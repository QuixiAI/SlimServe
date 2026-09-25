#!/usr/bin/env bash
set -euo pipefail

cd /home/eric/SlimServe
run_dir=perf/results/2026-09-25/qwen36-gap-diagnostic
mkdir -p "$run_dir"
server_pid=
restore() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  sudo systemctl start slimserve-affine-king.service
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "Restored healthy production service" | tee -a "$run_dir/diagnostic.log"
      return
    fi
    sleep 2
  done
  echo "ERROR: production service failed to recover" | tee -a "$run_dir/diagnostic.log" >&2
  return 1
}
trap restore EXIT

sudo systemctl stop slimserve-affine-king.service
/home/eric/.venv/bin/python -m slimserve.cli qwen36-nvfp4-1 --no-spec --serve --host 127.0.0.1 --port 8001 -y >"$run_dir/base-nospec-serve.log" 2>&1 &
server_pid=$!
for _ in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "ERROR: diagnostic server exited" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8001/health >/dev/null
echo "Diagnostic server healthy" | tee -a "$run_dir/diagnostic.log"
for concurrency in 1 8; do
  /home/eric/.venv/bin/python benchmarks/benchmark_dsv4_exact.py \
    --model /home/eric/models/Qwen3.6-35B-A3B-NVFP4 \
    --served-model-name Qwen3.6-35B-A3B \
    --source perf/results/2026-09-24/qwen36-5090-profile/source.md \
    --url http://127.0.0.1:8001/v1/completions \
    --concurrency "$concurrency" --input-tokens 1000 --output-tokens 2000 \
    --temperature 1 --top-p 0.95 --top-k 20 --seed 42 \
    --prompt-offset $((concurrency * 1000)) --allow-no-spec \
    >"$run_dir/base-nospec-c$concurrency.json"
  echo "Completed c$concurrency" | tee -a "$run_dir/diagnostic.log"
done
