#!/usr/bin/env bash
set -euo pipefail
cd /home/eric/SlimServe
run_dir=perf/results/2026-09-25/affine-online-fp8-dense
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
VLLM_MODELOPT_NVFP4_ONLINE_FP8_DENSE=1 \
  /home/eric/.venv/bin/python -m slimserve.cli affine-king-nvfp4-1 \
  --no-spec --serve --host 127.0.0.1 --port 8001 -y \
  >"$run_dir/serve.log" 2>&1 &
server_pid=$!
for _ in $(seq 1 240); do
  if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "ERROR: candidate exited before health" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8001/health >/dev/null
echo "Candidate healthy"
for concurrency in 1 8 64; do
  output_tokens=2000
  if [[ "$concurrency" == 64 ]]; then output_tokens=256; fi
  prompt_offset=$((concurrency * 1000))
  if [[ "$concurrency" == 64 ]]; then prompt_offset=0; fi
  /home/eric/.venv/bin/python benchmarks/benchmark_dsv4_exact.py \
    --model /home/eric/models/affine-king-r21-grpo5-s75-vision-NVFP4 \
    --served-model-name QuixiAI/affine-king-r21-grpo5-s75-vision-NVFP4 \
    --source perf/results/2026-09-24/qwen36-5090-profile/source.md \
    --url http://127.0.0.1:8001/v1/completions \
    --concurrency "$concurrency" --input-tokens 1000 \
    --output-tokens "$output_tokens" \
    --temperature 1 --top-p 0.95 --top-k 20 --seed 42 \
    --prompt-offset "$prompt_offset" --allow-no-spec \
    >"$run_dir/c$concurrency.json"
  echo "Completed c$concurrency"
done

/home/eric/.venv/bin/python tools/check_affine_online_fp8_dense.py \
  --url http://127.0.0.1:8001 \
  --output "$run_dir/candidate-canaries"

/home/eric/.venv/bin/python benchmarks/benchmark_dsv4_exact.py \
  --model /home/eric/models/affine-king-r21-grpo5-s75-vision-NVFP4 \
  --served-model-name QuixiAI/affine-king-r21-grpo5-s75-vision-NVFP4 \
  --source perf/results/2026-09-24/qwen36-5090-profile/source.md \
  --url http://127.0.0.1:8001/v1/completions \
  --concurrency 1 --repeat-source --input-tokens 261000 \
  --output-tokens 128 --warmup-output-tokens 0 \
  --temperature 1 --top-p 0.95 --top-k 20 --seed 42 \
  --allow-no-spec >"$run_dir/fullctx.json"
echo "Completed near-full-context request"
