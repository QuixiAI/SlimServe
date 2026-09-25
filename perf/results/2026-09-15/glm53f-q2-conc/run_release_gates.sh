#!/bin/bash
# Release gate, one boot: the three single-stream pins, the c=1 400-out
# concurrent pin, then concurrent_gate_pin.py at c=4/8/16 (1000-out).
# Usage: run_release_gates.sh <tag> [--pin]   (SKIP_BOOT=1: use the server
# already booting/serving)
set -u
D=/Users/seangherardi/Code/slimserve/SlimServe-glm53f/perf/results/2026-09-15/glm53f-q2-conc
PY=/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python
TAG=$1; shift
if [ -z "${SKIP_BOOT:-}" ]; then
  bash "$D/stop_server.sh" >/dev/null 2>&1
  $PY "$D/boot_env.py" "$D/$TAG"
fi
t0=$(date +%s)
for i in $(seq 1 120); do
  sleep 5
  if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 200; then break; fi
  if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then echo "server exited"; tail -5 "$D/$TAG/boot.log"; exit 1; fi
done
echo "READY ~$(( $(date +%s) - t0 ))s"
echo "== single-stream pins"; bash "$D/../../2026-09-11/glm53f-q2-baseline/gate.sh" "$D/$TAG/pins1"
echo "== 400-out c=1"; bash "$D/concurrent_gate.sh" "$D/$TAG" 1
echo "== concurrent pin gate"; $PY "$D/concurrent_gate_pin.py" "$D/$TAG/conc" "$@"
bash "$D/stop_server.sh"
echo "[$(date +%T)] ALL DONE"
