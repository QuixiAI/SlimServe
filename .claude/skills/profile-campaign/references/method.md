# Method: gates, notebook, rubric, PR description

This is the measurement contract every campaign phase is held to. It
condenses `perf/perf.md`, the campaign sections of HANDOFF.md and the
retrospectives; where they differ, the repo's `perf/perf.md` wins.

## The harness

`benchmarks/benchmark_dsv4_exact.py` is the exact-token harness for every
platform: it builds prompts of an exact token count from a frozen source
(`perf/results/harness_assets/m2_source.txt`; never regenerate it, the pins
depend on its bytes), returns per-request sha256 of the completion, and
reports `exact: true` only when every request honored the token counts.
`--concurrency N` builds N distinct prompts at strided offsets and returns N
shas. `aggregate_output_tps` includes each request's prefill: compare it
only across arms with the same prompt shape.

Standard legs (adapt the names, keep the shapes):

| leg | shape | purpose |
| --- | --- | --- |
| 8tok | 1000 in / 8 out, offset 1 | fast sha (prefill numerics) |
| off1-2000 | 1000 in / 2000 out, offset 1 | the decode pin; step ms = (wall - prefill) / steps |
| 2500x64 | 2500 in / 64 out, offset 0 | long-context anchor (multi-chunk prefill) |
| cN | 1000 in / 300 out at c4, c8, c16 (up to the record's max_num_seqs) | concurrency; aggregate and per-request decode |
| seeded | temperature 1.0 / top-p 0.95 / top-k 20 / seed 42 | the model's recommended sampling; the number the PR quotes |
| deep | 16k / 32k / 128k / native context needles | context correctness and TTFT |

Greedy legs are sha-pinned; the seeded arm is the throughput of record.
Never quote a single-offset tok/s as a result (trajectory lottery): means
over offsets, or the paired step ms.

Speculation is on in every gate once it is registered; acceptance (accepted
tokens per step, drafts per cycle) is recorded alongside tok/s, and the step
rate (tok/s divided by mean accepted length) is what a kernel change is
judged on.

## Gate classes

- **Bit-exact**: identical shas and identical spec counters, plus a liveness
  proof that the new path actually ran (a breadcrumb, a PSO / kernel name in
  the trace, a kill-switch reversion that changes something). Identical shas
  after a change that should alter numerics means "not engaged", not "safe".
- **ULP**: a change to summation order or fast-math visibility. Requires
  determinism x2, coherent text, paired step ms, means over at least five
  offsets, and a kill-switch reversion that reproduces the prior sha. The
  new sha becomes the pin, recorded with its teacher-forced or per-layer
  check.
- **Re-pin**: a deliberate numerics change (fp32 routing, a different
  drafter). Recorded with needles and the parity check; every pin that
  rolled is listed with the reason.

Other profiles that share a kernel body are anchors: re-gate them bit-exact
after any rebuild of a shared artifact and record the consecutive count.

## Correctness references, in order of strength

1. Teacher-forced comparison against the baseline engine's own logprobs
   (argmax agreement and mean |d logprob| over short and long prompts).
2. Per-layer cosine against a reference implementation (transformers on
   CPU for the first layers; the optimized path on another platform).
3. Planted-string needles at 1.5k / 4k / 12k, then 16k / 32k / 128k / native.
4. The model's own quality fixture if the quant author published one.
5. Canaries: a text request, a tool call, an image request where the model
   has vision.

Validation is proportionate: one needle per context tier per retained
numerics change, not four per wave.

## Boot protocol

Per platform (see `platform-ops.md`). In every case: boot detached, wait for
`/health`, send a tiny primer, then one ~1000-token single-chunk request
with decode, only then multi-chunk prefill or concurrency. Kill on the
platform's poison signatures (GPU command-buffer errors, NaN storms,
scheduler wedges); a poisoned boot is lost, never measured.

## Notebook entry (`perf/optimization_status.md`)

New entries go at the top of the file, one per experiment, in this shape:

```markdown
## YYYY-MM-DD - <model> <quant> on <platform>: <wave id> <short name> - <one-line verdict with numbers>

- Status: retained | rejected | parked | in progress | blocked
- Scope: model / profile / kernel / platform
- Baseline: the pins and step ms this was measured against, with their run dir
- Hypothesis: what changes, why it should help, expected ms/step, correctness contract
- Change: files and kernels, the kill switch, how another profile is kept off it
- Correctness: gate class, shas, parity numbers, needles
- Results: paired numbers (before / after), acceptance if speculative, GB/s or launches if kernel-level
- Decision: retained | rejected | parked, with the reason in one sentence
- Raw artifacts: perf/results/YYYY-MM-DD/<run-id>/ (never /tmp)
```

Rejections are one paragraph and the code leaves the tree in the same
commit. A "found while verifying" defect gets its own entry.

`perf/baseline_status.md` holds only stable snapshots: the bar (the baseline
engine on this box, its version, its best config, the exact commands), the
N0 bring-up pins, and each re-pin with the wave that rolled it. Summarize;
link the raw run dir.

## Rubric: retain, reject, park

Retain only if all hold:

1. Correctness: parity test passes at the serving shapes; gate class
   satisfied; canaries pass; `exact: true` in every harness cell; no
   degeneration under speculation.
2. Throughput: c1 and the concurrent cells within noise or better between
   like-state boots; the profiler pair shows the mechanism the hypothesis
   named (fewer launches, a shorter class, a smaller gap). Faster for a
   reason other than the hypothesis is a finding, not a retained change.
3. Prefill: not regressed at the standard lengths for any change that
   touches it.
4. Code: one kill switch or none; no second copy of a serving path; no
   diagnostic-only branch left behind; a test exists for the shape gate
   (which M / N / K / dtype takes the new path).
5. Notebook: the entry exists with raw artifact paths before the commit.

Reject after the time box when any fails; park when neutral; never retain on
a best-of-boots number. A sweep is at most one hour of microbench over at
most three parameters after research chose the design; never a serving
sweep of configurations.

## Concurrency bar

Batching must earn its keep. The operator's standing bar (2026-09-15): at
least 2x the c=1 aggregate at c=4 and at least 3x at c=16, with the record
sized for 16-32 concurrent requests where memory allows. Below that, the
record is not finished, however good single-stream is, unless the notebook
proves the ceiling from distinct-expert or bandwidth arithmetic. Speculation
that loses at batch is turned off by a batch-size schedule
(`num_speculative_tokens_per_batch_size`), not removed.

## PR description template

The PR description opens with the bar. Everything after it is the ledger.

```markdown
# <Model> <quant> on <platform>: <what was built> - <ratio> vs <bar engine> decode, <ratio> prefill

## The bar (same machine, clean memory, <bar engine> <version> at its best config vs `slimserve <id>`)

| workload | <bar engine> | SlimServe | ratio |
| --- | --- | --- | --- |
| decode c1, 1000 in / 2000 out (seeded) | | | |
| decode c1 without a drafter | | | |
| decode c4 / c8 / c16 aggregate | | | |
| prefill 1000 / 2048 / 2500 tokens | | | |
| TTFT 32k / 128k (cold, warm) | | | |

Correctness: <teacher-forced / parity numbers>, needles <tiers>, pins <shas>.

## What this PR adds
<kernels, serving path, adapter, profile, tests - one bullet each, with the mechanism>

## Retained / rejected ledger
<one line per wave, pointing at the notebook>

## Decisions of record
<operator decisions and their reasons, e.g. KV cache not quantized, drafter choice>

## Anchors
<other profiles re-gated bit-exact, with the consecutive count>

## Open items on the record
<what the record's notes state as not yet qualified>

## Tests
<files, counts, on which build>

## Kernel port
<QuixiCore branch or "follow-up">
```

No email addresses, no host names, no scratchpad paths, no mention of how
the work was produced. Under ~100 files per PR; split by layer if larger.
