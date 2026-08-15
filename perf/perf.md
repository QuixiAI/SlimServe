# SlimServe Performance Handbook

This is the operating guide for optimizing SlimServe inference. The goal is to
run a disciplined loop: study the strongest platform implementation, form a
bottleneck hypothesis, measure a clean baseline, run controlled experiments,
keep only verified wins, and record enough detail that the next pass starts
from evidence instead of memory.

The running notebook is `perf/optimization_status.md`. Stable baseline snapshots
live in `perf/baseline_status.md`. Raw benchmark output belongs under
`perf/results/YYYY-MM-DD/<run-id>/` and is not committed.

## Principles

- Correctness comes first. A performance change is not a win until it passes the
  relevant correctness checks and exact-token serving harnesses.
- Measure on realistic serving shapes. Toy kernels are useful for isolation, but
  retained decisions must improve the target model on the target hardware.
- Change one meaningful factor per experiment: layout, fusion boundary, launch
  geometry, staging, quant decode, routing threshold, or scheduling policy.
- Record rejected ideas. A rejected measurement is useful if it prevents the
  same detour later.
- Compare platforms deliberately. When ROCm, Metal, or CUDA has an optimized
  path, study its algorithm, layout, and fusion choices before implementing the
  next platform.

## Required Measurement Record

Every recorded experiment should include:

- Date, git commit or working-tree label, and author/context.
- Model, quant, profile name, tensor parallel degree, and exact serving command.
- Hardware: GPU model/count, driver, CUDA/HIP/ROCm version, CPU, host memory,
  and any power/clock constraints known.
- Kernel or end-to-end path under test, including public Python/C++/CUDA/HIP
  entry points.
- Baseline implementation and target implementation.
- Prompt workload: input tokens, output tokens, concurrency, sampling settings,
  chat/completion endpoint, and prompt source.
- Warmup, timed iterations, repeat count, median and spread, plus raw output
  location.
- Correctness criteria and observed max error or exact-token validation result.
- Throughput: tokens/s for serving, and GB/s or TFLOP/s for microbenchmarks
  when those estimates are meaningful.
- Decision: retained, rejected, or follow-up required, with the reason.

## DSV4 0731 A100 Loop

For DeepSeek V4 Flash 0731 on Ampere A100, the current target is the native
fused MoE path:

```text
IQ2_XXS gate/up -> SwiGLU -> Q2_K down -> weighted reduce
```

The implementation must support vLLM's combined `[expert, gate|up, packed]`
GGUF layout or split/aligned artifacts. Do not record dequant or standalone
Q2_K GEMV as production wins; those can be diagnostics only.

The ROCm DeepSeek V4 Flash 0731 implementation is the primary cross-platform
baseline. DS4, QuixiCore, llama.cpp, and vLLM Marlin should be treated as
reference material for scheduling, quant layouts, tensor-core usage, and
benchmark shape selection, not as endpoints.

## Canonical Serving Benchmarks

Use SlimServe profiles so downloads, parser settings, DSpark/TurboQuant env,
KV dtype, and CUDA graph settings stay on the real path:

```bash
.venv/bin/python -m slimserve.cli dsv4-2 --quant IQ2_XXS --serve --host 127.0.0.1 --port 8000 -y
.venv/bin/python -m slimserve.cli dsv4-4 --quant IQ2_XXS --serve --host 127.0.0.1 --port 8000 -y
```

Use the exact-token harness for comparable throughput:

```bash
.venv/bin/python benchmarks/benchmark_dsv4_exact.py \
  --model /home/ubuntu/models/antirez-deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --url http://127.0.0.1:8000/v1/completions \
  --concurrency 1 \
  --input-tokens 1000 \
  --output-tokens 2000 \
  --prompt-overhead 9
```

Repeat at concurrency 8 unless the experiment is explicitly a single-request
latency probe. Store full logs under `perf/results/`.

## Profilers (Metal)

Two opt-in instruments ship with the Metal serving path; both are inert
unless enabled and add no cost when off.

- `VLLM_QC_PHASE_PROF=1` — sync-bracketed step-phase split (target forward /
  sample / drafter propose) from `vllm/v1/worker/metal_phaseprof.py`. Dumps
  to `/tmp/phaseprof_<pid>.txt`. The brackets serialize the pipeline, so
  absolute step time is inflated; trust the split, not the totals, and use
  xctrace (`metal-gpu-intervals` + encoder-label join) for per-kernel
  attribution.
- `VLLM_SYNCPROF=1` — sync-point census from
  `vllm/v1/worker/metal_syncprof.py`: wraps the host-blocking torch entry
  points (`item`/`tolist`/`cpu`/`synchronize`/`AsyncOutput.get_output`) with
  call-site attribution, plus an optional command-buffer census. Dumps to
  `/tmp/syncprof_<pid>.txt` on an interval (`VLLM_SYNCPROF_INTERVAL`,
  default 30 s); `VLLM_SYNCPROF_TARGETS` subsets the wrapped names.

## Notebook Format

Each notebook entry should use this shape:

```markdown
## YYYY-MM-DD - Short Experiment Name

- Status: retained | rejected | in progress | blocked
- Scope: model/profile/kernel/platform
- Baseline:
- Hypothesis:
- Change:
- Correctness:
- Results:
- Decision:
- Raw artifacts:
```

Keep `perf/optimization_status.md` concise but complete. Link raw logs instead
of pasting long output.
