# GLM53 bounded prompt-score qualification

Status: CPU and isolated CUDA exactness/memory gates pass; source freeze ended.
The helper is not installed in serving. No quant, profile or native changes.

## Hypothesis and scope

The completed indexer-serving series logged eight recovered 4,718,592,000-byte
allocation warnings in every arm. This is the 2 MiB-rounded size of a
[7616,154880] FP32 matrix. Source inspection suggests full-chunk prompt score
conversion/log_softmax; the actual warning has not been stack-attributed.

Keep the complete vocabulary projection and TP gathering unchanged, then bound
only row-independent scoring temporaries to 1024 rows using the existing Sampler
conversion/log_softmax, top-k and inclusive-rank operations. Requested outputs
are not scratch and can still be large for full-vocabulary requests. The existing
639-row score-journal path remains unchunked. No arithmetic substitute or fallback.

## Prescribed isolated CUDA process v1

Exactly one GPU0 process, one fresh private Inductor/Triton namespace. No model
server, native build, power/clock changes, simultaneous GPU work or retries. Stop
on any failure, preserve the partial record, and prescribe any follow-up separately.
GPU scope 16 GiB host RAM/swap0; CPU checks 8 GiB/swap0. Freeze listed probe
sources from launch until process exit. GPU release is checked after exit.

The script fixes 60 cases in order: BF16 rows1/639/1024/1025/2051/7616, vocabulary
154880, four score modes and k0/k5; then FP16/FP32 at1025 rows with raw modes/k5;
then all-equal BF16 ties at1025 rows, vocabulary154880/k5 and17/k17, four modes.
Offset rows and padded vocabulary test noncontiguous inputs and guards. Actual
Sampler methods, including compiled CUDA inclusive rank, are mandatory.

Each case runs control then chunked, one full untimed warmup per arm, then three
measurements per arm. Exact dtype/shape/value equality for scores, token IDs and
ranks is required for every call; input/guards unchanged. CUDA-event and wall
times plus peak incremental allocated bytes are recorded, not serving TPS.
After the7616-row raw-logprobs/k0 measurements, separately profile both warmed
arms with allocation history and operator shapes. These isolated allocation
stacks do not, by themselves, attribute the live-server warning.

```bash
systemd-run --user --scope --unit=glm53-prompt-scores-gpu-v1 -p MemoryMax=16G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-10/prompt-scores-cache-v1/inductor TRITON_CACHE_DIR=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-10/prompt-scores-cache-v1/triton .venv/bin/python -m benchmarks.kernels.check_glm53_prompt_scores --output perf/results/2026-09-10/prompt-scores-gpu-v1
```

## Required later gate

Only after local exactness and memory qualification, wire the helper into the
runner with small-request/journal semantics intact. Then prescribe a real fixed
`glm53-nvfp4-4`/`rtx6000`, recipe v1 control/candidate/return comparison with
unchanged quality gates, exact per-token score comparison, text/image canaries,
cold exact-token throughput and allocation-warning census. No serving promotion
based on the isolated probe. No serving process is prescribed by this document yet.

## CPU result

184 tests pass in5.98s: helper tests and existing score-journal tests, CPU emulator
only. Raw `perf/results/2026-09-10/runtime-control/prompt-score-chunks-cpu-v1.xml`.

## Completed isolated result

One prescribed process, all60 cases and480 checked outputs pass exactly; three
timings per arm/case. No retry or replacement. Hardware UUID/driver/600W unchanged,
GPUs released after exit. Raw `perf/results/2026-09-10/prompt-scores-gpu-v1/`;
`result.json` SHA68b82a83a8a7bc0ba8cc9b4ae349e6de6a0882f280e51edfec41aa99993cc271.
Source hashes in that completed receipt precede this result documentation.

At7616 rows/BF16/raw-logprobs/k0, incremental peak allocation falls from
9,437,184,000 to1,270,996,480 bytes (86.5% less). CUDA median20.122 to19.597ms;
all three control20.093..20.140 and chunked19.584..19.606ms. This is a warmed,
fixed-order isolated scoring measurement, not a decode/prefill/serving speed win.
Small1025-row shapes have expected extra-launch overhead (~3..6%); this does not
affect the unchanged<=1024-row runner fast path planned for integration.

The separate allocation traces contain exactly two large control allocations,
4,718,264,320 requested bytes each: FP32 conversion (`aten::_to_copy`) and
`aten::_log_softmax`, with Python/C++ stacks through Sampler.compute_logprobs.
The chunked trace instead contains16 score allocations, maximum634,388,480 bytes
(1024 rows), with277,544,960-byte tail allocations. Rounded allocator accounting
explains the4,718,592,000-byte control blocks. This establishes isolated operation
attribution, not an actual live-server OOM stack. Serving integration and warning
elimination remain pending; profile/default/baseline are unchanged.
