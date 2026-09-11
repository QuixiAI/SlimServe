# Same-live-input prompt scoring diagnostic v1

Status: CPU-qualified 357 tests/29.95s; one fresh serving start prescribed below.
This is an additional causal diagnostic, NOT a replacement rollout, revised
model-quality gate, or default-promotion protocol. The failed rollout stays failed.

## Fixed target and hypothesis

`glm53-nvfp4-4`/`rtx6000`, recipe `glm53-redhatai-nvfp4-fp8-kda-tp4-v1`, TP4
Marlin; BF16 activations/KV/vocabulary head, BF16fn1/TC0, native-order0. No EP,
speculation, normalization intervention, forced AOT loading or copied compiler
artifacts. The existing qualified chunked scorer itself remains byte-unchanged.

Question: on the SAME full logits produced by real long prompts under production
ordering, does bounded scoring preserve every selected token, score bit and rank?
Historical control variation cannot confound this identical-input comparison.

## Implementation and exact gates

Opt-in `SLIMSERVE_GLM53_PROMPT_SCORE_SHADOW=<result>/shadow` requires chunks1 and
rejects other GLM53 diagnostics. CLI/campaign validate the fixed recipe and compiler
policy. The runner computes the model projection/TP gather only once. For each
score chunk above1024 rows, the shadow computes existing full scoring, releases its
large scratch, then invokes the unchanged bounded scorer on the same tensors.

SHA256 of all logical input bytes is checked before/after each scorer, including
target IDs. Host hash staging is bounded to32MiB. Outputs require matching shapes/
dtypes, finite scores and exact byte hashes for token IDs, scores and ranks. All
target IDs, scores and ranks are retained for the HTTP join. The process saves
begin/complete/failure events; interrupted/failed records cannot pass the auditor.

Small chunks keep the original scorer and are explicitly marked UNPAIRED. They
do not count toward chunked-path coverage. Worker scope is BF16 logits,154880
vocabulary,1..8192 rows, raw_logprobs, k0..5; max512 scoring calls/rank. CPU fixtures
cover k0/5, short/threshold/long chunks, offsets and deliberate input/output damage.
The live protocol below requests k0 only; it does not qualify live top-k5 behavior.

The CPU-only independent auditor requires four rank journals with exact source
receipts. It joins EVERY request/chunk to the actual complete HTTP prompt scores,
IDs and ranks, requires contiguous complete coverage of all56 unique requests,
checks all three output hashes against their recorded values, and demands observed
paired shapes reaching at least7616 rows. Extra/missing/duplicated ranks, requests,
chunks, changed offsets, bytes, coverage labels or truncated events fail closed.

## Prescribed workload and failure policy

Exactly ONE start with an independently empty compile cache. Real profile discovery
and serving health, text/image canaries, three cold exact1000/300 timing repeats
at c1/c8/c16, one32-window/24-needle-candidate quality pass, cold32K/128K prefill
with warmups and three repetitions. No journal-driven model/AOT interventions.

This run executes both full and bounded scoring and synchronizing host hashes;
ALL timings are diagnostic-only, not performance baselines or memory savings.
Recovered reference allocation warnings are expected and must remain recorded.

The unchanged historical aggregate/window/needle comparison is reported separately,
including failures. It is NOT silently converted to a pass and does not answer the
different question of same-input scorer parity. A complete diagnostic report never
promotes chunking, ordering, model quality, reproducibility, or a speed baseline.

Controller/audit8GiB, serving150GiB, swap0; no competing GPU work or native builds.
Sources/extra auditor/protocol/native/prompt receipts freeze after commit through
terminal report. Exactly one attempt; no retry, replacement or resumed failed run.
Keep full response/log/journal/cache inventory. Verify GPU release before edits.

After commit:

```bash
systemd-run --user --scope --unit=glm53-prompt-shadow-v1-controller -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.run_glm53_prompt_score_shadow --output perf/results/2026-09-10/prompt-score-shadow-v1
```

Raw controller `result.json`, `serve.log`, campaign, independent fresh cache and
per-rank `shadow/*.jsonl` all live under that single new directory.

CPU history: `runtime-control/prompt-shadow-cpu-v1.xml`325 pass/14.01s;
`prompt-shadow-cpu-v2.xml`357 pass/29.95s under2026-09-10. The second suite includes
all-request synthetic HTTP-join failure tests, one-attempt controller failure/
interruption preservation, historical-quality-failure separation, existing rollout,
runner, sampler and benchmark-exclusion regressions. No GPU run consumed by tests.
