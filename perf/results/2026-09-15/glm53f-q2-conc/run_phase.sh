#!/bin/bash
# Phase-profiled c=16 1000-out split: boot with the profiler (overlap
# regions off: the profiler cannot sync inside them), snapshot the
# cumulative dump after the c=1 warm-up leg and after the c=16 leg, diff.
# Usage: run_phase.sh <tag> <conc> [KEY=VAL ...]
set -u
D=/Users/seangherardi/Code/slimserve/SlimServe-glm53f/perf/results/2026-09-15/glm53f-q2-conc
PY=/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python
TAG=$1; CONC=$2; shift 2; OUT=$D/$TAG; mkdir -p "$OUT"
bash "$D/stop_server.sh" >/dev/null 2>&1
$PY "$D/boot_env.py" "$OUT" VLLM_QC_PHASE_PROF=1 VLLM_METAL_MOE_OVERLAP=0 VLLM_METAL_SHARD_OVERLAP=0 VLLM_METAL_MOE_ROUTER_OVERLAP=0 "$@"
t0=$(date +%s)
for i in $(seq 1 120); do
  sleep 5
  if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 200; then break; fi
done
echo "READY ~$(( $(date +%s) - t0 ))s"
PDIR=$(ls -td /var/folders/*/*/T/qc_phaseprof_* 2>/dev/null | head -1)   # the EngineCore print is block-buffered in boot.log
echo "dumps: $PDIR"
bash "$D/concurrent_gate.sh" "$OUT/warm" 1
sleep 3; cp "$PDIR"/phaseprof_*.txt "$OUT/phase_base.txt"
OUT_TOKENS=1000 bash "$D/concurrent_gate.sh" "$OUT/c$CONC" $CONC
sleep 3; cp "$PDIR"/phaseprof_*.txt "$OUT/phase_after_c16.txt"
bash "$D/stop_server.sh"
$PY "$D/phase_diff.py" "$OUT/phase_base.txt" "$OUT/phase_after_c16.txt" | tee "$OUT/phase_diff.txt"
echo "[$(date +%T)] PHASE DONE"
