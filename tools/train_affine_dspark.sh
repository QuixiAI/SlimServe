#!/usr/bin/env bash
# Capture Affine King hidden states and fine-tune its published Qwen3.6 DSpark head.
set -euo pipefail

repo=/home/eric/SlimServe
venv=/home/eric/.venv/bin
speculators=/home/eric/speculators
model=/home/eric/models/affine-king-r21-grpo5-s75-vision-NVFP4
run="$repo/perf/results/2026-09-25/affine-dspark-train"
corpus="$run/full-corpus.jsonl"
prepared="$run/full-prepared"
hidden="$run/full-hidden-states"
checkpoints="$run/full-checkpoints"
extract_pid=

if [[ ! -s "$corpus" ]]; then
  echo "Missing filtered on-policy corpus: $corpus" >&2
  exit 1
fi

"$venv/speculators" prepare-data \
  --model "$model" --data "$corpus" --output "$prepared" \
  --seq-length 4096 --minimum-valid-tokens 256 \
  >"$run/full-prepare.log" 2>&1

cleanup() {
  if [[ -n "$extract_pid" ]]; then
    kill -TERM -- "-$extract_pid" 2>/dev/null || true
    wait "$extract_pid" 2>/dev/null || true
  fi
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

sudo -n systemctl stop slimserve-affine-king.service
mkdir -p "$hidden"

cd "$speculators"
setsid env VLLM_USE_V2_MODEL_RUNNER=0 \
  LD_LIBRARY_PATH="/usr/local/cuda-13.3/compat:${LD_LIBRARY_PATH:-}" \
  "$venv/python" scripts/launch_vllm.py train \
  "$model" --target-layer-ids 2 10 20 30 37 40 \
  --hidden-states-path "$hidden" -- \
  --host 127.0.0.1 --port 8001 --max-model-len 8192 \
  --gpu-memory-utilization 0.82 --kv-cache-dtype fp8 \
  --cpu-offload-gb 4 --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --moe-backend marlin --renderer-num-workers 1 \
  >"$run/full-extract-serve.log" 2>&1 &
extract_pid=$!

ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8001/health >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$extract_pid" 2>/dev/null; then
    break
  fi
  sleep 3
done
if [[ "$ready" != 1 ]]; then
  echo "Hidden-state capture server did not reach health" >&2
  exit 1
fi

"$venv/speculators" generate-offline-data \
  --model "$model" --endpoint http://127.0.0.1:8001/v1 \
  --preprocessed-data "$prepared" --output "$hidden" \
  --concurrency 4 --request-timeout 300 --validate-outputs --fail-on-error \
  >"$run/full-extract.log" 2>&1

kill -TERM -- "-$extract_pid" 2>/dev/null || true
wait "$extract_pid" 2>/dev/null || true
extract_pid=

"$venv/speculators" train \
  --verifier-name-or-path "$model" --data-path "$prepared" \
  --save-path "$checkpoints" \
  --from-pretrained /home/eric/models/affine-king-dspark-warmstart \
  --speculator-type dspark --target-layer-ids 2 10 20 30 37 \
  --draft-attn-impl sdpa --epochs 5 --total-seq-len 4096 \
  --max-anchors 64 --lr 1e-5 --loss-fn '{"ce":0.1,"tv":0.9}' \
  --hidden-states-path "$hidden" --on-missing raise --save-best \
  --log-freq 50 \
  >"$run/full-train.log" 2>&1
