#!/bin/bash
# One boot: 400-out gates at c=1/8/16 (pins + curve), then the decode-
# dominated 1000-out at c=8/16. Usage: run_gates_full.sh <tag> [KEY=VAL ...]
set -u
D=/Users/seangherardi/Code/slimserve/SlimServe-glm53f/perf/results/2026-09-15/glm53f-q2-conc
PY=/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python
TAG=$1; shift
bash "$D/stop_server.sh" >/dev/null 2>&1
$PY "$D/boot_env.py" "$D/$TAG" "$@"
t0=$(date +%s)
for i in $(seq 1 120); do
  sleep 5
  if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 200; then break; fi
done
echo "READY ~$(( $(date +%s) - t0 ))s"
echo "== 400-out"; bash "$D/concurrent_gate.sh" "$D/$TAG" ${CONC400:-1 8 16}
echo "== ${LONG_TOKENS:-1000}-out"; OUT_TOKENS=${LONG_TOKENS:-1000} bash "$D/concurrent_gate.sh" "$D/${TAG}_long" ${CONC1000:-8 16}
bash "$D/stop_server.sh"
echo "[$(date +%T)] ALL DONE"
