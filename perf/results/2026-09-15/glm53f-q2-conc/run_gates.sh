#!/bin/bash
# boot (profile env + KEY=VAL overrides) -> wait READY -> gates -> stop.
# Usage: run_gates.sh <outdir> "<conc list>" [KEY=VAL ...]
set -u
D=/Users/seangherardi/Code/slimserve/SlimServe-glm53f/perf/results/2026-09-15/glm53f-q2-conc
PY=/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python
OUT=$1; CONC=$2; shift 2
mkdir -p "$OUT"
bash "$D/stop_server.sh" >/dev/null 2>&1
$PY "$D/boot_env.py" "$OUT" "$@"
t0=$(date +%s)
for i in $(seq 1 120); do
  sleep 5
  if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 200; then break; fi
done
echo "READY ~$(( $(date +%s) - t0 ))s"
bash "$D/concurrent_gate.sh" "$OUT" $CONC
bash "$D/stop_server.sh"
echo "[$(date +%T)] ALL DONE"
