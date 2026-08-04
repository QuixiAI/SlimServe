#!/bin/bash
# Benchmark a model via vllm serve + curl
# Usage: bench_pr1_serve.sh <model_name> <label>
set -e

MODEL="$1"
LABEL="$2"
PORT=8199

echo ""
echo "============================================================"
echo "Benchmarking: $LABEL"
echo "Model: $MODEL"
echo "============================================================"

# Start server in background
VLLM_LOGGING_LEVEL=WARNING ~/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len 512 \
    --gpu-memory-utilization 0.85 \
    --port $PORT \
    --disable-log-requests \
    2>&1 &
SERVER_PID=$!

echo "Waiting for server to start..."
for i in $(seq 1 360); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server failed to start within 360s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Warmup (2 runs to capture CUDA graphs)
echo "Warming up..."
for w in 1 2 3; do
    curl -s http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"Hello world\", \"max_tokens\": 50, \"temperature\": 0}" \
        > /dev/null
done

PROMPT="Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves."

echo "Benchmarking (3 runs, 200 tokens each)..."
TOTAL_TPS=0
for run in 1 2 3; do
    START=$(date +%s%N)
    RESULT=$(curl -s http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"max_tokens\": 200, \"min_tokens\": 200, \"temperature\": 0}")
    END=$(date +%s%N)

    TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
    ELAPSED_MS=$(( (END - START) / 1000000 ))
    TPS=$(python3 -c "print(f'{$TOKENS / ($ELAPSED_MS / 1000):.1f}')")
    echo "  Run $run: $TOKENS tokens in ${ELAPSED_MS}ms = $TPS tok/s"
    TOTAL_TPS=$(python3 -c "print(f'{$TOTAL_TPS + $TOKENS / ($ELAPSED_MS / 1000):.1f}')")
done

AVG=$(python3 -c "print(f'{$TOTAL_TPS / 3:.1f}')")
echo ""
echo "  Average: $AVG tok/s"
echo "============================================================"

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
