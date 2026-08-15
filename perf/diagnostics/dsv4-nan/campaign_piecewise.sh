#!/bin/bash
# Multi-boot arm: TP4 with PIECEWISE capture-32 graphs (the TP2 profile's
# graph mode, never observed to degenerate) instead of FULL_DECODE_ONLY
# capture-64. Everything else matches the production config. 6 boots,
# 2x c11 trigger runs each, dual tripwires.
set -u
SC=/tmp/claude-1001/-home-ubuntu-SlimServe/1c975788-d502-4dff-93f8-4258ce64e11d/scratchpad/dualbench
MODEL=/home/ubuntu/models/antirez-deepseek-v4-gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf
for BOOT in 1 2 3 4 5 6; do
  NAME=pw$BOOT
  sudo rm -f /var/log/SlimServe/ss-$NAME.log
  sudo systemd-run --unit=ss-$NAME --uid=ubuntu --gid=ubuntu \
    -p WorkingDirectory=/home/ubuntu/SlimServe \
    -E PYTHONPATH=/home/ubuntu/SlimServe -E PYTHONUNBUFFERED=1 -E CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -E VLLM_NAN_WATCH=1 -E VLLM_DSV4_TOPK_VALIDATE=1 -E VLLM_DSV4_ALIGNED_Q8=1 -E VLLM_DSV4_MHC_SCHEDULE=async \
    -p StandardOutput=append:/var/log/SlimServe/ss-$NAME.log -p StandardError=append:/var/log/SlimServe/ss-$NAME.log \
    /home/ubuntu/.local/SlimServe-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 127.0.0.1 --port 1804$BOOT --tensor-parallel-size 4 --kv-cache-dtype fp8 --block-size 256 \
    --attention-config '{"sparse_mla_force_mqa": true}' \
    --compilation-config '{"cudagraph_mode": "PIECEWISE", "max_cudagraph_capture_size": 32}' \
    --trust-remote-code --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 \
    --served-model-name DeepSeek-v4-Flash-0731 --gpu-memory-utilization 0.78 --max-model-len 262144 \
    --speculative-config '{"model": "/home/ubuntu/models/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF/DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf", "method": "dspark", "num_speculative_tokens": 5, "quantization": "gguf", "attention_backend": "TURBOQUANT", "kv_cache_dtype": "turboquant_k8v4"}'
  for i in $(seq 1 100); do
    c=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:1804$BOOT/health 2>/dev/null)
    [ "$c" = "200" ] && break
    sleep 10
  done
  for i in 1 2; do
    /home/ubuntu/SlimServe/.venv/bin/python /home/ubuntu/SlimServe/benchmarks/benchmark_dsv4_exact.py \
      --model "$MODEL" --served-model-name DeepSeek-v4-Flash-0731 --source $SC/r5-b-c16.txt \
      --url http://127.0.0.1:1804$BOOT/v1/completions --concurrency 11 --input-tokens 1000 --output-tokens 2000 \
      --prompt-offset $(( i * 55 )) > $SC/$NAME-$i.json 2> $SC/$NAME-$i.log || true
  done
  NAN=$(sudo grep -c 'NAN_WATCH:' /var/log/SlimServe/ss-$NAME.log 2>/dev/null || true)
  VAL=$(sudo grep -c 'TOPK_VALIDATE:' /var/log/SlimServe/ss-$NAME.log 2>/dev/null || true)
  DEG=$(python3 -c "
import json
t=0
for i in (1,2):
    try:
        d=json.load(open('$SC/$NAME-$i.json'))
        t+=sum(1 for r in d['chars_per_token'] if r<0.5)
    except Exception: pass
print(t)")
  echo "PW BOOT $BOOT: degenerate_reqs=$DEG nan_lines=${NAN:-0} validate_lines=${VAL:-0}"
  sudo systemctl stop ss-$NAME 2>/dev/null
  sleep 15
done
echo "PW CAMPAIGN DONE"
