#!/usr/bin/env bash
# Compare the temporarily registered Affine DSpark candidate with the live baseline.
set -euo pipefail

repo=/home/eric/SlimServe
run="$repo/perf/results/2026-09-25/affine-dspark-train"
venv=/home/eric/.venv/bin
profile="$repo/slimserve/profiles.json"
mode=${EVAL_MODE:-fixed}
[[ "$mode" == fixed || "$mode" == dynamic ]] || { echo "Invalid evaluation mode" >&2; exit 2; }
default_label=full-candidate
[[ "$mode" == dynamic ]] && default_label=full-dynamic-candidate
label=${EVAL_LABEL:-$default_label}
[[ "$label" =~ ^[a-z0-9-]+$ ]] || { echo "Invalid evaluation label" >&2; exit 2; }
candidate_pid=

"$venv/python" - <<'PY'
from pathlib import Path

draft = Path('/home/eric/models/affine-king-dspark-full')
expected = {'config.json': 2039, 'config.py': 2342, 'model.safetensors': 883339898}
for name, size in expected.items():
    path = draft / name
    if not path.is_file() or path.stat().st_size != size:
        raise SystemExit(f'Missing or wrong-size trained draft file: {path}')
PY

stage_profile() {
  "$venv/python" - "$profile" "$mode" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
mode = sys.argv[2]
text = path.read_text()
base = '''"local_dir": "Qwen3.6-35B-A3B-speculator.dspark",
        "download_strategy": "hf_hub",
        "files": [
          {"path": "config.json", "bytes": 2003},
          {"path": "config.py", "bytes": 2342},
          {"path": "model.safetensors", "bytes": 1900458850}'''
candidate = '''"local_dir": "affine-king-dspark-full",
        "download_strategy": "hf_hub",
        "files": [
          {"path": "config.json", "bytes": 2039},
          {"path": "config.py", "bytes": 2342},
          {"path": "model.safetensors", "bytes": 883339898}'''
if text.count(base) != 1:
    raise SystemExit("Base profile block missing or ambiguous; refusing replacement")
text = text.replace(base, candidate, 1)
if mode == "dynamic":
    before = '"num_speculative_tokens": 4,\n          "kv_cache_dtype": "fp8"'
    after = '"num_speculative_tokens": 4,\n          "num_speculative_tokens_per_batch_size": [[1, 4, 4], [5, 64, 0]],\n          "kv_cache_dtype": "fp8"'
    if text.count(before) != 1:
        raise SystemExit("Affine DSpark engine block missing or ambiguous")
    text = text.replace(before, after, 1)
json.loads(text)
path.write_text(text)
PY
}

restore_profile() {
  "$venv/python" - "$profile" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
candidate = '''"local_dir": "affine-king-dspark-full",
        "download_strategy": "hf_hub",
        "files": [
          {"path": "config.json", "bytes": 2039},
          {"path": "config.py", "bytes": 2342},
          {"path": "model.safetensors", "bytes": 883339898}'''
base = '''"local_dir": "Qwen3.6-35B-A3B-speculator.dspark",
        "download_strategy": "hf_hub",
        "files": [
          {"path": "config.json", "bytes": 2003},
          {"path": "config.py", "bytes": 2342},
          {"path": "model.safetensors", "bytes": 1900458850}'''
if text.count(candidate) == 1:
    text = text.replace(candidate, base, 1)
elif text.count(base) != 1:
    raise SystemExit("Affine DSpark profile block missing or ambiguous")
dynamic = '          "num_speculative_tokens_per_batch_size": [[1, 4, 4], [5, 64, 0]],\n'
if dynamic in text:
    if text.count(dynamic) != 1:
        raise SystemExit("Dynamic schedule ambiguous; refusing replacement")
    text = text.replace(dynamic, "", 1)
path.write_text(text)
PY
}

cleanup() {
  if [[ -n "$candidate_pid" ]]; then
    kill -TERM -- "-$candidate_pid" 2>/dev/null || true
    wait "$candidate_pid" 2>/dev/null || true
  fi
  restore_profile
  sudo -n systemctl start slimserve-affine-king.service
  for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "Production Affine King service restored and healthy"
      return 0
    fi
    sleep 3
  done
  echo "Production Affine King service did not regain health" >&2
  return 1
}
trap cleanup EXIT

stage_profile
"$venv/python" - "$profile" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text()
assert text.count('"local_dir": "affine-king-dspark-full"') == 1
assert text.count('{"path": "model.safetensors", "bytes": 883339898}') == 1
PY

sudo -n systemctl stop slimserve-affine-king.service
cd "$repo"
setsid env VLLM_KV_TIER_VERIFY=1 "$venv/python" -m slimserve.cli \
  affine-king-nvfp4-1 --spec --serve --host 127.0.0.1 --port 8001 -y \
  >"$run/$label-serve.log" 2>&1 &
candidate_pid=$!

ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$candidate_pid" 2>/dev/null; then
    break
  fi
  sleep 3
done
if [[ "$ready" != 1 ]]; then
  echo "Candidate SlimServe profile did not reach health" >&2
  exit 1
fi

benchmark() {
  local concurrency=$1 output_tokens=$2 offset=$3
  "$venv/python" benchmarks/benchmark_dsv4_exact.py \
    --model /home/eric/models/affine-king-r21-grpo5-s75-vision-NVFP4 \
    --served-model-name QuixiAI/affine-king-r21-grpo5-s75-vision-NVFP4 \
    --source perf/results/2026-09-24/qwen36-5090-profile/source.md \
    --url http://127.0.0.1:8001/v1/completions \
    --concurrency "$concurrency" --input-tokens 1000 \
    --output-tokens "$output_tokens" --temperature 1 --top-p 0.95 \
    --top-k 20 --seed 42 --prompt-offset "$offset" \
    >"$run/$label-c$concurrency.json" \
    2>"$run/$label-c$concurrency.stderr"
}

benchmark 1 2000 1000
benchmark 8 2000 2000
benchmark 64 256 0

"$venv/python" - "$run" "$label" <<'PY'
from pathlib import Path
import json
import sys

run = Path(sys.argv[1])
label = sys.argv[2]
summary = {}
for concurrency in (1, 8, 64):
    base = json.loads((run / f"fresh-nodraft-c{concurrency}.json").read_text())
    draft = json.loads((run / f"{label}-c{concurrency}.json").read_text())
    summary[str(concurrency)] = {
        "baseline_tps": base["aggregate_output_tps"],
        "candidate_tps": draft["aggregate_output_tps"],
        "speedup": draft["aggregate_output_tps"] / base["aggregate_output_tps"],
        "exact": draft["exact"],
        "draft_tokens": draft["spec_decode_draft_tokens"],
        "accepted_tokens": draft["spec_decode_accepted_tokens"],
    }
(run / f"{label}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
