#!/usr/bin/env bash
set -euo pipefail
cd /home/eric/SlimServe
run_dir=perf/results/2026-09-25/qwen36-gap-profile
mkdir -p "$run_dir"
server_pid=
restore() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
  sudo systemctl start slimserve-affine-king.service
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "Restored healthy production service"
      return
    fi
    sleep 2
  done
  echo "ERROR: production service failed to recover" >&2
  return 1
}
trap restore EXIT
sudo systemctl stop slimserve-affine-king.service

for variant in affine base; do
  if [[ "$variant" == affine ]]; then
    profile=affine-king-nvfp4-1
    model=/home/eric/models/affine-king-r21-grpo5-s75-vision-NVFP4
    served=QuixiAI/affine-king-r21-grpo5-s75-vision-NVFP4
    spec_arg=--no-spec
  else
    profile=qwen36-nvfp4-1
    model=/home/eric/models/Qwen3.6-35B-A3B-NVFP4
    served=Qwen3.6-35B-A3B
    spec_arg=--no-spec
  fi
  mkdir -p "$run_dir/$variant"
  /home/eric/.venv/bin/python -m slimserve.cli "$profile" "$spec_arg" \
    --serve --host 127.0.0.1 --port 8001 -y \
    --torch-profile-dir "$run_dir/$variant" >"$run_dir/$variant/serve.log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 240); do
    if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "ERROR: $variant server exited" >&2
      exit 1
    fi
    sleep 2
  done
  curl -fsS http://127.0.0.1:8001/health >/dev/null
  curl -fsS -X POST http://127.0.0.1:8001/start_profile >/dev/null
  /home/eric/.venv/bin/python benchmarks/benchmark_dsv4_exact.py \
    --model "$model" --served-model-name "$served" \
    --source perf/results/2026-09-24/qwen36-5090-profile/source.md \
    --url http://127.0.0.1:8001/v1/completions \
    --concurrency 1 --input-tokens 1000 --output-tokens 64 \
    --temperature 1 --top-p 0.95 --top-k 20 --seed 42 \
    --prompt-offset 5000 --allow-no-spec >"$run_dir/$variant/c1-trace-request.json"
  curl -fsS -X POST http://127.0.0.1:8001/stop_profile >/dev/null
  echo "Traced $variant"
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=
done
