#!/bin/bash
# Concurrent exact-token gate set for glm53f-q2-1.
#
# Same protocol as perf/results/2026-09-11/glm53f-q2-baseline/gate.sh (the
# three single-stream pins), extended along --concurrency. The harness builds
# N DISTINCT prompts at strided source offsets, returns a per-request sha
# list, and reports "exact" only when every request returned the requested
# prompt and completion token counts.
#
# Output tokens are 400 (not the smoke check's 96) so prefill and scheduler
# ramp stop dominating the window; --warmup-output-tokens 1 primes each
# prompt so the timed run reads a warm prefix cache. Per-request decode is
# reported alongside aggregate: aggregate alone hides the latency each
# request pays for the batch.
#
# Usage: concurrent_gate.sh <outdir> [concurrency ...]
set -u
WT=/Users/seangherardi/Code/slimserve/SlimServe-glm53f
PY=/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python
GGUF=$HOME/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf
OUT=${1:?usage: concurrent_gate.sh <outdir> [concurrency ...]}
shift
CONC=${*:-1 2 4 8}
mkdir -p "$OUT"; cd "$WT"; export PYTHONPATH=$WT

H="$PY benchmarks/benchmark_dsv4_exact.py --model $GGUF --served-model-name GLM-5.3-Flash --source perf/results/harness_assets/m2_source.txt --url http://127.0.0.1:8000/v1/completions --metrics-url none --allow-no-spec --temperature 0 --warmup-output-tokens 1"

for c in $CONC; do
  name="c$c"
  echo "[$(date +%T)] $name"
  $H --concurrency "$c" --input-tokens 1000 --output-tokens "${OUT_TOKENS:-400}" --prompt-offset 1 \
     > "$OUT/$name.json" 2> "$OUT/$name.err"
  grep -iE 'did not honor|Traceback|error' "$OUT/$name.err" | head -2
done
"$PY" "$WT/perf/results/2026-09-15/glm53f-q2-conc/summarize.py" "$OUT"
echo "[$(date +%T)] CONCURRENT GATES DONE"
