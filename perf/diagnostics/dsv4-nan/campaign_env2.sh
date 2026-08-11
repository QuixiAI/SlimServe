#!/bin/bash
# Fast env-discriminator arms for the rare NaN events. The step-6 warmup
# event fires on most boots, so a short c11 run (300 output tokens) per
# boot gives the event-rate signal in ~4 min/boot. Three boots per arm.
# Arms: baseline (profile env), aligned-q8 off, prequant-attn off,
# aux-streams off.
set -u
SC=/tmp/claude-1001/-home-ubuntu-SlimServe/1c975788-d502-4dff-93f8-4258ce64e11d/scratchpad/dualbench
MODEL=/home/ubuntu/models/antirez-deepseek-v4-gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf

run_boot() {
  local NAME=$1 PORT=$2; shift 2
  sudo rm -f /var/log/SlimServe/ss-$NAME.log
  sudo systemd-run --unit=ss-$NAME --uid=ubuntu --gid=ubuntu \
    -p WorkingDirectory=/home/ubuntu/SlimServe \
    -E PYTHONPATH=/home/ubuntu/SlimServe -E PYTHONUNBUFFERED=1 -E CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -E VLLM_NAN_WATCH=1 -E VLLM_NAN_WATCH_LAYERS=1 -E VLLM_DSV4_TOPK_VALIDATE=1 "$@" \
    -p StandardOutput=append:/var/log/SlimServe/ss-$NAME.log -p StandardError=append:/var/log/SlimServe/ss-$NAME.log \
    /home/ubuntu/.local/SlimServe-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 127.0.0.1 --port $PORT --tensor-parallel-size 4 --kv-cache-dtype fp8 --block-size 256 \
    --attention-config '{"sparse_mla_force_mqa": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "max_cudagraph_capture_size": 64}' \
    --trust-remote-code --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 \
    --served-model-name DeepSeek-v4-Flash-0731 --gpu-memory-utilization 0.78 --max-model-len 262144 \
    --speculative-config '{"model": "/home/ubuntu/models/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF/DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf", "method": "dspark", "num_speculative_tokens": 5, "quantization": "gguf", "attention_backend": "TURBOQUANT", "kv_cache_dtype": "turboquant_k8v4"}'
  for i in $(seq 1 100); do
    c=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$c" = "200" ] && break
    sleep 10
  done
  for R in 1 2; do
    /home/ubuntu/SlimServe/.venv/bin/python /home/ubuntu/SlimServe/benchmarks/benchmark_dsv4_exact.py \
      --model "$MODEL" --served-model-name DeepSeek-v4-Flash-0731 --source $SC/r5-b-c16.txt \
      --url http://127.0.0.1:$PORT/v1/completions --concurrency 11 --input-tokens 1000 --output-tokens 2000 \
      --prompt-offset $(( R * 55 )) > $SC/env-$NAME-$R.json 2> $SC/env-$NAME-$R.log || true
  done
  local NAN VAL
  NAN=$(sudo grep -c 'NAN_WATCH:' /var/log/SlimServe/ss-$NAME.log 2>/dev/null || true)
  VAL=$(sudo grep -c 'TOPK_VALIDATE:' /var/log/SlimServe/ss-$NAME.log 2>/dev/null || true)
  echo "ENVARM $NAME: nan_lines=${NAN:-0} validate_lines=${VAL:-0}"
  sudo systemctl stop ss-$NAME 2>/dev/null
  sleep 12
}

# Wait for the pilot campaign to release the GPUs before starting.
while pgrep -f "campaign_env.sh" > /dev/null 2>&1; do sleep 20; done
sleep 30
for B in 1 2 3 4; do
  run_boot fb$B 1811$B -E VLLM_DSV4_ALIGNED_Q8=1 -E VLLM_DSV4_MHC_SCHEDULE=async
  run_boot fq$B 1812$B -E VLLM_DSV4_ALIGNED_Q8=0 -E VLLM_DSV4_MHC_SCHEDULE=async
done
echo "ENV2 CAMPAIGN DONE"
