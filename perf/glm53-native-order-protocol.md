# GLM53 native ordering: normal execution qualification

Frozen before the first launch, 2026-09-09. This is a candidate qualification,
not a new baseline or default promotion. No quant, arithmetic or native binary
changes relative to the completed stable-align diagnostic on 3e1dac08f.

## Hypothesis and gates

The qualified source-order routing/alignment and ordered pool selector should
preserve every quality score without launch serialization or tensor journals.
Use one start, not a retry-until-fast loop. Run all three quality passes even
if their score vectors differ; the offline audit must retain every difference.
Serving errors stop this run and remain failures. No replacement starts.

- Registered `glm53-nvfp4-4` / RTX6000 / TP4 / no EP / no speculation.
- Recipe `glm53-redhatai-nvfp4-fp8-kda-tp4-v1`, BF16 activations/KV/head,
  mHC BF16 fn storage ON, mHC prefill tensor cores OFF.
- Only `SLIMSERVE_GLM53_NATIVE_ORDER=1` selects ordering. All legacy ordering
  controls and all score/model/MoE/index journals absent. No launch blocking.
- Default JIT logging, not verbose; request metrics ON for cold-cache checks.
  No profiler, CUDA trace, routing capture, long-context timing or tensor dump.
- One full 1000-in/300-out warmup per c1/c8/c16, three timed repetitions each:
  25 warmup and 75 timed requests, every request uncached and exact-token.
- Text and image canaries; then three full quality passes, each 32 x 128 text
  scores plus six four-candidate needle contrasts at 1K/8K/32K, all uncached.
- Require every text score and every needle-token score to repeat exactly.
  Compare all nine pairings against the frozen stable-align diagnostic's three
  passes. No tolerance adjustment; a mismatch is evidence for further isolation.
- No tensor-journal parity claim is possible with journals off. TPS is recorded
  as a sanity/qualification observation, not compared as a new competitive win.
- Sources and native binaries frozen through serving and prescribed audits;
  no builds, edits to serving/client/native files, commits or competing GPU work.
  Preserve raw logs, partial outputs, warnings and slow samples. Verify owned
  process teardown and driver release before resuming development.

Native SHA256s:

```
QC   39b302f041bb846712b396f84100aefefcb332ed3fd04bd787853fa43bfdb31c
core fe4a7c2a3c2c03cc8f725528e40aead70f2570cdbb9bb1481d4874c7e6427639
MoE  1093b8a4ca7cb308d4ebff254ab502d7b01d65eb86423b2640c8f8a4bff4ac1a
```

## Command

Run from the repository after committing the policy and passing CPU/GPU wiring
checks. Check GPU ownership immediately before launch. This command creates a
new output directory; do not change its name to hide a failed start.

```bash
set -o pipefail
systemd-run --user --scope --unit=glm53-native-order-quality \
  -p MemoryMax=150G -p MemorySwapMax=0 \
  env -u CUDA_LAUNCH_BLOCKING -u NCCL_P2P_DISABLE \
  -u SLIMSERVE_GLM53_CANONICAL_MOE -u SLIMSERVE_GLM53_STABLE_ROUTE \
  -u SLIMSERVE_GLM53_STABLE_ALIGN -u SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER \
  -u SLIMSERVE_GLM53_CANONICAL_INDEX_TIES -u SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED \
  -u SLIMSERVE_GLM53_MODEL_JOURNAL -u SLIMSERVE_GLM53_MOE_JOURNAL \
  -u SLIMSERVE_GLM53_SCORE_JOURNAL -u SLIMSERVE_GLM53_INDEX_JOURNAL \
  CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda-13.0 \
  SLIMSERVE_CACHE=/raid/weights \
  VLLM_CACHE_ROOT=/home/tiny/.local/scratch/slimserve-glm53/vllm-cache \
  OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  VLLM_GLM5_MHC_BF16_FN=1 VLLM_GLM5_MHC_PREFILL_TC=0 \
  SLIMSERVE_GLM53_NATIVE_ORDER=1 \
  .venv/bin/python benchmarks/benchmark_glm53_campaign.py \
  --profile glm53-nvfp4-4 \
  --source /home/tiny/.local/scratch/slimserve-glm53/prompt-source.txt \
  --output perf/results/2026-09-09/native-order-quality \
  --boots 1 --repeats 3 --concurrency 1 8 16 \
  --input-tokens 1000 --output-tokens 300 --cold-prefix \
  --quality --quality-repeats 3 \
  2>&1 | tee perf/results/2026-09-09/runtime-control/native-order-quality-launch.log
```

Read-only CPU audits use an 8 GiB/no-swap scope, reconstruct quality from raw
HTTP responses, check every request/token/cache value and all source/native
receipts. Audit outputs belong under `runtime-control/native-order-*`. The
control is `perf/results/2026-09-09/stable-align-quality-diagnostic/`, summary
SHA256 `7a262eee2748daec06d90f55238a362bd0bf4af3ebe4731f5273d98ca5915222`.
Only plan difference allowed: the control's explicit verbose JIT option is
absent. The environment differs by the stated observer/ordering controls.
