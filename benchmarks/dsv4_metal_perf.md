# DeepSeek V4 Flash 0731 Metal throughput worklog

This log follows the measurement loop in
`~/QuixiCore/QuixiCore-Metal/perf/perf.md`: establish a correctness-checked
baseline, profile one bottleneck, change one variable, and retain only repeated
material wins.

## Fixed workload

- Host: Apple M5 Max, 128 GiB unified memory, macOS 26.5.2.
- Target: the local 97,591,747,456-byte DeepSeek V4 Flash 0731 hybrid GGUF.
- Prompt source: natural text from `perf.md`, with distinct windows for each
  concurrent request.
- Request: exactly 1,000 API input tokens and 2,000 generated tokens.
- Sampling: greedy (`temperature=0`), `ignore_eos=true`.
- Thinking: enabled through the model's default DeepSeek completion template.
- Concurrency: measured separately at 1 and 8.
- Correctness gate: every API response must report the exact requested input
  and output token counts. The harness exits nonzero otherwise.

The completion adapter adds nine template tokens. The harness therefore builds
991-token raw prompts (`--prompt-overhead 9`) and validates that the server
reports exactly 1,000 input tokens.

## Harness

`benchmarks/benchmark_dsv4_exact.py` tokenizes locally from the GGUF, creates
distinct exact-length prompts, sends concurrent OpenAI completion requests,
and reports both aggregate output throughput and request latency.

Canonical invocation:

```sh
.venv/bin/python benchmarks/benchmark_dsv4_exact.py \
  --model "$MODEL" \
  --source ~/QuixiCore/QuixiCore-Metal/perf/perf.md \
  --url http://127.0.0.1:18080/v1/completions \
  --concurrency 8 \
  --input-tokens 1000 \
  --output-tokens 2000 \
  --prompt-overhead 9
```

## Engine selection

The in-tree vLLM Metal port can load the model, but its copied MPS residency is
96.77 GiB before KV/cache overhead. Profiling a single token drove system swap
from about 12 GiB to about 29 GiB and stalled in per-expert MoE execution. It
also still lacks the real two-cache sparse MLA path required by this model.

The DS4 Metal backend maps the same local GGUF without copying the weights and
plans 91.18 GiB total at a 3,008-token context (90.88 GiB model, 0.28 GiB KV,
0.02 GiB other buffers). It already implements the DeepSeek V4 sparse attention
and routed-expert kernels, so it is the viable Metal backend for SlimServe.

## Baselines

### Concurrency 1

Server:

```sh
./ds4-server --metal -m "$MODEL" --ctx 3008 --tokens 2000 \
  --host 127.0.0.1 --port 18080
```

Exact result:

```json
{
  "aggregate_output_tps": 33.68418388593356,
  "completion_tokens": [2000],
  "concurrency": 1,
  "exact": true,
  "prompt_tokens": [1000],
  "wall_seconds": 59.37504695891403
}
```

Server timing was 3.236 seconds for prefill and 56.134 seconds for decode
(35.63 decode tok/s averaged by the server).

### Concurrency 8

Server:

```sh
./ds4-server --metal -m "$MODEL" --ctx 3008 --tokens 2000 \
  --batched-session 8 --host 127.0.0.1 --port 18080
```

Exact result:

```json
{
  "aggregate_output_tps": 32.82139652399084,
  "completion_tokens": [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],
  "concurrency": 8,
  "exact": true,
  "prompt_tokens": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
  "request_latency_mean_seconds": 485.1643081197253,
  "request_latency_median_seconds": 487.0281273954897,
  "wall_seconds": 487.4868742499966
}
```

Steady-state per-request decode was about 4.5--4.6 tok/s. Aggregate decode
throughput was therefore only similar to concurrency 1, showing that the
batch path was saturating weight traffic without enough cross-request reuse.

## Experiments

### Native routed-expert row batching

Observation: the existing single-device Metal session batch combined Q/KV and
shared-expert operations, but invoked routed MoE separately for every session.
The routed experts dominate decode and their weights are the best opportunity
for reuse across concurrent rows.

Change: gather each session's normalized FFN row and selected experts, run the
existing expert-sorted multi-row Metal MoE kernels once per layer, then feed
each routed row into the existing fused shared-down/HC expansion. The path has
the opt-out `DS4_METAL_SESSION_BATCH_MOE=0` for clean A/B tests.

Correctness smoke, 8 x (1,000 input / 100 output): exact for every request. A
first implementation that tried to reuse the multi-device shared-FFN helper
failed immediately and was rejected; the replacement retains the established
single-device shared-FFN arithmetic.

Full exact result:

```json
{
  "aggregate_output_tps": 34.35343719172318,
  "completion_tokens": [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],
  "concurrency": 8,
  "exact": true,
  "prompt_tokens": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
  "request_latency_mean_seconds": 463.94291786447866,
  "request_latency_median_seconds": 465.2872944374685,
  "wall_seconds": 465.74669983400963
}
```

Compared with baseline: aggregate throughput +4.67%, wall time -4.46%.
Retained.

### Routed-expert batching plus Q/KV row batching

The first retained implementation disabled the prior Q/KV row batch because
both paths originally ended at different layer boundaries. A new decode phase
stops after the externally batched Q/KV projections and router, allowing both
optimizations to compose without repeating attention or FFN work.

Full exact result:

```json
{
  "aggregate_output_tps": 35.34653665838861,
  "completion_tokens": [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],
  "concurrency": 8,
  "exact": true,
  "prompt_tokens": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
  "request_latency_mean_seconds": 450.42937511474884,
  "request_latency_median_seconds": 452.21594322950114,
  "wall_seconds": 452.6610387499677
}
```

Compared with the original baseline: aggregate throughput +7.69%, wall time
-7.14%. Compared with routed-expert batching alone: aggregate throughput
+2.89%, wall time -2.81%. This is provisionally retained because it is just
below the isolated 3% threshold; the next paired-QKV experiment repeats the
same composed path while removing half of its projection dispatches.

#### Rejected: paired Q/KV projection dispatch

The exact paired Q8 projection kernel handles one row at a time. Replacing the
two existing eight-row 2D projection dispatches with eight paired one-row
dispatches passed the 8 x (1,000 input / 100 output) correctness smoke, but
regressed wall time from about 64.88 to 65.29 seconds. Rejected and restored
the two multi-row projections.

### Complete mixed prefills instead of interleaving 128-token slices

Observation: with the default `--mixed-prefill-quantum 128`, the first request
started decode while the other prompts were repeatedly resumed in 128-token
slices. The final prompt did not finish until about 43.4 seconds. Setting
`--mixed-prefill-quantum 2048` lets every 1,000-token prompt finish in one
mixed-prefill pass; all eight prompts completed by 22.4 seconds.

Full exact result:

```json
{
  "aggregate_output_tps": 36.78408331255119,
  "completion_tokens": [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],
  "concurrency": 8,
  "exact": true,
  "prompt_tokens": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
  "request_latency_mean_seconds": 434.40681008323736,
  "request_latency_median_seconds": 434.5155297500314,
  "wall_seconds": 434.97074166696984
}
```

Compared with routed-MoE plus Q/KV batching at the default mixed quantum:
aggregate throughput +4.07%, wall time -3.91%. Compared with the original
baseline: aggregate throughput +12.07%, wall time -10.77%. Retained; 2,048 is
the new c8 scheduler reference.

### Rejected count-8 routed-MoE fusions

Three exact 8 x (1,000 input / 300 output) screens used the retained kernels
and 2,048-token mixed prefill quantum:

| Variant | Aggregate tok/s | Wall seconds | Versus control |
|---|---:|---:|---:|
| Control | 30.7858 | 77.9581 | -- |
| Extend tiny pair/SwiGLU from <=5 to 8 rows | 28.6159 | 83.8694 | -7.05% |
| Extend direct down-sum from <=4 to 8 rows | 29.0222 | 82.6955 | -5.73% |

Both candidates passed exact token-count checks but regressed materially. The
tiny pair kernels remain appropriate for DSpark's small verification block;
the ordinary multi-row serving path is better at eight rows. Both experiments
were rejected and their threshold switches removed. Their combination was not
run because both independent components were already negative.

### Correctness re-audit: completion digests

The harness now records a SHA-256 digest of each returned completion, in
addition to exact prompt/completion token counts. This exposed a failure that
the earlier length-only checks could not detect: identical c8 runs through the
external routed-MoE layer split produced different completion digests. Six of
eight responses differed from the unsplit reference, and two responses also
varied between repeated runs. Disabling external routed-MoE batching restored
all eight stable digests across repeated runs. Disabling all native session
batching produced those same eight digests, showing that the established
shared/QKV row path is deterministic and the router boundary is the source of
the drift.

The routed-MoE result and its composed Q/KV result above are therefore
invalidated and rejected despite their throughput gains. The scheduler result
remains independently applicable, but its final c8 number must be remeasured
on the stable path. The external routed-MoE phase and its environment switch
were removed entirely; no path crosses the unsafe router boundary.

#### Rejected: batched Q-b projection

Batching the remaining Q-b projection required another layer-resume phase
after Q normalization/RoPE and KV storage. The batched projection and its
explicit command boundaries completed, but the layer-0 continuation failed
the server's exact-token correctness check. The phase, temporary diagnostics,
and opt-out were removed. The retained implementation continues to batch only
Q-a and KV-a, using the established per-session continuation for Q-b and
attention.

#### Rejected: canonical per-row routed fallback after external router

Replacing the fast multi-row routed kernel with the canonical one-token MoE
kernel did not restore reproducibility while keeping the external router
boundary. Repeated 20-token screens changed all eight completion digests, so
the issue is the split layer-state contract rather than only the fused batch
kernel. This fallback was removed.

### Response correctness: invalid UTF-8 sanitation

A stable full c8 run generated all requested tokens, but one completion
contained tokenizer bytes that were not a valid UTF-8 sequence. The server's
JSON escaper previously copied every byte above ASCII unchanged, producing an
invalid OpenAI JSON response. The serializer now validates UTF-8, preserves
valid multi-byte sequences, and emits `\uFFFD` for invalid bytes. A focused
server unit test covers both invalid-byte replacement and valid four-byte UTF-8
preservation. The complete DS4 server unit suite passes.

### Stable c8 reference

With external routed-MoE disabled, Q/KV row batching retained, the UTF-8 fix,
and `--mixed-prefill-quantum 2048`, the full required benchmark is:

```json
{
  "aggregate_output_tps": 35.350183356299944,
  "completion_tokens": [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000],
  "concurrency": 8,
  "exact": true,
  "prompt_tokens": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
  "request_latency_mean_seconds": 452.03757348950603,
  "request_latency_median_seconds": 452.15209558350034,
  "wall_seconds": 452.61434258299414
}
```

Compared with the original c8 baseline: aggregate throughput +7.70%, wall
time -7.15%. This is the current correctness-qualified c8 reference. The
eight full-completion SHA-256 digests are recorded in the raw harness output;
future c8 candidates must reproduce them or explain a deliberate, validated
numerical change.

### C1 Q8 matvec dispatch sweep

Two repeated 1 x (1,000 input / 300 output) controls averaged 27.72 aggregate
tok/s (the number includes prefill). The default Q8 matvec uses four SIMD
groups and two output rows per dispatch. Every alternate SIMD-group count was
slower in isolated screens:

| SIMD groups | Aggregate tok/s | Versus control |
|---:|---:|---:|
| 1 | 26.998 | -2.61% |
| 2 | 27.300 | -1.52% |
| 3 | 27.464 | -0.93% |
| 4 (control) | 27.721 | -- |
| 5 | 27.361 | -1.30% |
| 6 | 27.267 | -1.64% |
| 7 | 27.034 | -2.48% |
| 8 | 27.003 | -2.59% |

Exact Q8 model views averaged 27.70 tok/s and were neutral (-0.1%). Four
output rows reached 28.89 and 29.23 tok/s in two screens, but identical runs
returned different completion digests. It is rejected as nondeterministic.
The default four-group/two-row dispatch remains the correctness-qualified c1
choice.

### C1 decode command-buffer split sweep

The decode graph normally commits after every four transformer layers. Two
repeated 1 x (1,000 input / 300 output) controls averaged 27.72 aggregate
tok/s. Alternate split sizes all reproduced the control completion SHA-256
`88723fce4fa516f512681306b296d86824515435b72abe3d289f286fd15f9ea3`:

| Layers per command buffer | Aggregate tok/s | Versus control |
|---:|---:|---:|
| 0 (single buffer) | 27.140 | -2.10% |
| 1 | 27.511 | -0.76% |
| 2 | 27.785 | +0.23% |
| 4 (control) | 27.721 | -- |
| 8 | 27.680 | -0.15% |
| 16 | 27.501 | -0.79% |
| 32 | 27.340 | -1.37% |

The apparent 0.23% gain at two layers is below the retention threshold and
normal run-to-run noise. The existing four-layer split remains retained.

### C1 launch/fusion flag audit

The layer-40 stage profiler put the largest individual decode stages at roughly
0.30--0.38 ms (attention output and routed MoE), with the Q projection,
compressor/indexer projections, inverse RoPE, shared gate/up, and shared down
mostly in the 0.15--0.28 ms range. That profile motivated a sweep of existing
Metal fast-path flags on the exact 1 x (1,000 input / 300 output) screen. Every
candidate reproduced the control completion digest, but none cleared noise:

| Candidate | Aggregate tok/s |
|---|---:|
| Compressor APE-add fusion | 27.591 |
| Ratio-4 compressor pack fusion | 27.553 |
| Ratio-4 direct pool | 27.585 |
| Persistent zero attention mask | 27.557 |
| Shared-KV padding | 27.572 |
| HC RMS-scale projection | 27.543 |
| Zero-prefix prefill-mask cache | 27.599 |
| Q4 table auto | 27.557 |
| Q4 group size 24 | 27.548 |
| Q4 group size 6 | 27.577 |
| Q4 group size 8 | 27.568 |

The paired control was approximately 27.72 tok/s. The flags remain at their
established defaults. A GPU-only greedy top-1 sampler also preserved digest
`88723fce4fa516f512681306b296d86824515435b72abe3d289f286fd15f9ea3`
but left decode at about 39.4 tok/s and was removed.

### Rejected c8 attention-output batching

Batching both attention-output projections reached 20.8896 aggregate tok/s on
the 8 x (1,000 input / 100 output) screen versus 19.3066 for its paired
control, but four of eight completion digests changed. It was rejected for
numerical drift.

A stricter variant batched only the low-rank projection, then ran the canonical
per-session output projection plus HC expansion. It reproduced all eight
control digests exactly:

```text
6a179c2cf6ec903a5a1bf2848f630d2248604fce6440de7e79b1e3877ce03d1c
f20a4f75eaf704cac6fa678282a322cef64505f6fc00e7e72eca5ad669d6dab9
4f163a44495548786ba76871b6a6da0f1f59cc2537b300b37eeb45049142e03a
eedde41d2958ba5c40d4bf042aa2b0120aa6e691a4373f377ac092c0f60a763c
852f795625646578570b448723ec54e4102a32c9999b1ba274bb9d26df698bf5
000efb6310edb180bff7e070eaa98d70ca0d88881e170775761a6da879db6063
569adf57534df7f71b82d59e8ef5a613a920c76a9c526a281423c7a1c75d5fde
3fa1a47b10b7a0166686bb3e2cc70f652330b0a305f51f21b6dc07ecda63d0b2
```

The machine then entered a persistent degraded state: the strict candidate
measured 6.65 aggregate tok/s and its immediately paired unmodified control
measured only 7.13, versus the normal 19.3 control. A warmed 20-token rerun was
still incomplete after a minute. Those timings are invalid rather than an A/B
result. Because the candidate could not satisfy the measured >=3% retention
rule, all of its phases, API surface, and Metal implementation were removed.
No unproven attention-output batch path remains in the source.

### Other rejected c8 experiments

- A batched output head was exact but improved only about 2.57%, below the 3%
  retention threshold.
- Batched shared-down address dispatch improved about 0.6% and was removed.
- Decode coalescing sweeps kept the existing 2,000 microsecond value.
- Retaining Metal command buffers failed correctness; alternate command-buffer
  split counts were slower.
- Exact Q8 model views were neutral, while four-row Q8 decode was
  nondeterministic.
- Alternate Q8 NSG dispatches, routed count-8 fusions, and the tested output/HC
  fusion toggles were slower or numerically unstable.

The source tree was cleaned after every rejected experiment. The remaining DS4
changes are benchmark support/correctness (`ignore_eos`, UTF-8-safe JSON) and a
profiler-boundary fix; the retained throughput win is the c8 launch setting
`--mixed-prefill-quantum 2048` on top of the existing exact native row batches.

### C1 routed IQ2/Q2 geometry audit

The clean stage profile showed routed MoE as the largest repeated decode stage,
so the final kernel audit varied the fused IQ2 gate/up and Q2 down output rows
per SIMD group. All candidates preserved the 100-token control digest. Because
an unrelated memory-intensive process was still active, these are paired
relative screens in the degraded regime (normal clean decode is 35.6 tok/s;
these controls decoded at about 13.5 tok/s), not replacements for the clean
headline numbers.

| IQ2 rows | Q2 rows | Mean aggregate tok/s | Versus paired control |
|---:|---:|---:|---:|
| 4 | 4 | 6.132 | -- |
| 8 | 8 | 6.054 | -1.27% |
| 2 | 2 | 5.971 | -2.63% |
| 8 | 4 | 5.937 | -3.18% |
| 4 | 8 | 5.911 | -3.60% |

The four-row default is the best tested occupancy/register tradeoff. Replacing
the fused IQ2 kernel's threadgroup lookup-table copy and barrier with direct
constant-table reads was also exact but reduced mean aggregate throughput from
6.132 to 6.052 tok/s and decode from about 13.55 to 13.42 tok/s. It was
reverted. The active fused path remains pair-projection + SwiGLU followed by the
direct six-expert Q2 down-sum; there is no remaining decomposed launch to remove
without changing the cross-threadgroup dependency.

## Constraints

- No additional target models or DSpark drafters may be downloaded.
- The previously downloaded target is the only model used here.
- DSpark cannot be benchmarked until its already-approved local drafter exists.
  This work does not relabel DS4's built-in MTP mechanism as DSpark.
- TurboQuant/DSpark profile policy is tracked separately from these kernel
  measurements. Every SlimServe profile now emits DSpark with TurboQuant and
  names a blessed download URL, but this benchmark did not fetch or run those
  auxiliary weights.
