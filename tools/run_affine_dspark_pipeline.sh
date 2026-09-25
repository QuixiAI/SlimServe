#!/usr/bin/env bash
# Persistent, resumable on-policy regeneration followed by one-GPU DSpark training.
set -euo pipefail

repo=/home/eric/SlimServe
run="$repo/perf/results/2026-09-25/affine-dspark-train"
speculators=/home/eric/.venv/bin/speculators
model=QuixiAI/affine-king-r21-grpo5-s75-vision-NVFP4
endpoint=http://127.0.0.1:8000/v1/chat/completions
magpie_pid=
ultrachat_pid=

cleanup() {
  if [[ -n "$magpie_pid" ]]; then
    kill "$magpie_pid" 2>/dev/null || true
  fi
  if [[ -n "$ultrachat_pid" ]]; then
    kill "$ultrachat_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

regenerate() {
  local source=$1 seed=$2
  "$speculators" regenerate-responses \
    --endpoint "$endpoint" --model "$model" \
    --dataset "$run/$source-single-prompts.jsonl" \
    --limit 1000 --concurrency 24 --max-tokens 3072 \
    --sampling-params "{\"temperature\":1,\"top_p\":0.95,\"top_k\":20,\"seed\":$seed}" \
    --outfile "$run/$source-single-regen.jsonl" --resume \
    >"$run/$source-single-regen.log" 2>&1
}

regenerate magpie 42 &
magpie_pid=$!
regenerate ultrachat 43 &
ultrachat_pid=$!
wait "$magpie_pid"
magpie_pid=
wait "$ultrachat_pid"
ultrachat_pid=

for source in magpie ultrachat; do
  rows=$(wc -l <"$run/$source-single-regen.jsonl")
  errors=$(wc -l <"$run/$source-single-regen.errors.jsonl")
  if [[ "$rows" -ne 1000 || "$errors" -ne 0 ]]; then
    echo "$source regeneration incomplete: rows=$rows errors=$errors" >&2
    exit 1
  fi
done

/home/eric/.venv/bin/python "$repo/tools/prepare_affine_dspark_corpus.py" \
  --magpie "$run/magpie-single-regen.jsonl" \
  --ultrachat "$run/ultrachat-single-regen.jsonl" \
  --extra "$run/magpie-regen.jsonl" \
  --extra "$run/ultrachat-regen.jsonl" \
  --output "$run/full-corpus.jsonl" \
  >"$run/full-filter.log" 2>&1

/home/eric/.venv/bin/python - "$run/full-corpus.summary.json" <<'PY'
import json
import shutil
import sys

summary = json.load(open(sys.argv[1]))
tokens = sum(v["selected_tokens"] for v in summary.values() if isinstance(v, dict))
estimated_bytes = tokens * 6 * 2048 * 2  # six bf16 hidden-state layers
free_bytes = shutil.disk_usage(sys.argv[1]).free
print(f"selected={summary['total_selected']} tokens={tokens} ")
print(f"estimated_hidden_gib={estimated_bytes / 2**30:.1f} free_gib={free_bytes / 2**30:.1f}")
if summary["total_selected"] < 1000 or estimated_bytes + 25 * 2**30 > free_bytes:
    raise SystemExit("Corpus count or disk-headroom check failed")
PY

bash "$repo/tools/train_affine_dspark.sh"
touch "$run/full-trained.ok"
