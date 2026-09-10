# GLM53 indexer selective correction: isolated GPU protocol

Status: prospective diagnostic. Fixed recipe v1, TP4 RTX PRO 6000 SM120;
production attention, quant, defaults and model-quality floors stay unchanged.
This extra-launch diagnostic is not a speed optimization or serving integration.

Baseline: four source/config/binary-exact original attention bundles from the
completed rank-private probe. Their indexer LayerNorm128 one-BF16-ULP gate
failed (max5); the split kernel is not used here. The independent historical
failure remains recorded even if a new correction passes.

Hypothesis: actual BF16 output/bias identifies cancellation-prone elements;
FP64 recomputation can repair them without changing any unselected or sibling
values. The CPU screen in `glm53-attention-isolation.md` caught all retained
failures. Fixed trigger2^-12, chosen before GPU work; no threshold/config search.

## Candidate and boundaries

`benchmarks/kernels/glm53_indexer_correction.py` is opt-in probe code only:

- Original combo first, correction second, same stream.
- Output/bias detector in FP32, exact CPU-screen predicate. One warp per row;
  128 columns; num_stages1; compiler FP fusion disabled.
- Read input columns2048:2176 of `[rows,2336]`; recompute row mean, centered
  variance, inverse standard deviation, normalization and affine in FP64.
  Epsilon remains1e-6. Convert only the completed affine result to BF16.
- Store only flagged elements in the first128 columns of `[rows,256]`.
  Q1536, KV512, indexer gate, every unflagged value and input/weight/row/stride
  guards must remain unchanged. A guarded uint8 selection output records the
  actual detector decision; its overhead is included in diagnostic timing.

## Prescribed experiment

One CPU preparation, **one GPU process**, one terminal CPU audit. GPU0..3 run
sequentially, each using its own fresh rank-private cache. Exactly120 cases:
four ranks × rows(1,3,16,640,7616) × seeds(530901,530902) × magnitudes(.125,1,8).
Each case has the original and seed+100 phase, eager repetition and graph
replay with changed/restored input. Reproduce every original input/output hash.
No model weights beyond the four small layer11 BF16 norm tensors are loaded.

Correctness: exact GPU/CPU detector agreement; zero missed original failures;
corrected indexer <=1 BF16 ULP against the unchanged FP64 oracle; all non-target,
unflagged, input/weight/output/selection guards and graph checks pass. The final
model-quality gate is separate and NOT exercised by this probe.

Only after all120 cases pass, time rank0 rows(1,16,640,7616), seed530901,
magnitude1.0. Control then correction, three paired repetitions; ten eager
warmups and ten graph warmups per arm. Each timed graph contains32 calls.
CUDA events report microseconds per original bundle or bundle+correction,
including selection writes. No TPS, model performance or uninstrumented kernel
claim follows from these timings. No tuning, retries, replacement starts,
excluded cases or serving-cache writes. Stop and audit the first failed case.

## Resources and freeze

CPU preparation/tests/audit: user scope8GiB, swap0, CUDA hidden. GPU probe:
user scope16GiB, swap0, CUDA_VISIBLE_DEVICES=0,1,2,3; one Torch/OMP CPU thread.
Check no active compute workloads before starting. Record and compare GPU
identity/driver/power settings at entry and after process exit. No native build
or competing GPU work. Freeze code/protocol/compiler/receipts from preparation
through terminal audit; preserve all raw data and original5172-file inventory.

Commit the CPU-tested implementation before running these commands:

```bash
systemd-run --user --scope --unit=glm53-indexer-correction-prepare-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.check_glm53_indexer_correction prepare --manifest perf/results/2026-09-10/runtime-control/indexer-correction-manifest-v1.json
systemd-run --user --scope --unit=glm53-indexer-correction-gpu-v1 -p MemoryMax=16G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.check_glm53_indexer_correction run --manifest perf/results/2026-09-10/runtime-control/indexer-correction-manifest-v1.json --output perf/results/2026-09-10/indexer-correction-v1
systemd-run --user --scope --unit=glm53-indexer-correction-audit-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.check_glm53_indexer_correction audit --manifest perf/results/2026-09-10/runtime-control/indexer-correction-manifest-v1.json --output perf/results/2026-09-10/indexer-correction-v1
```

Preparation consumes pinned completed historical audits, not obsolete freeze
validators. It creates a new source freeze for this experiment. Audit failure
does not authorize another start: retain the failure and revise the hypothesis
or implementation under a new explicit protocol before any subsequent GPU run.
