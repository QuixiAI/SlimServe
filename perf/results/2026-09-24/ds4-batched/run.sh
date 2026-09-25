#!/bin/bash
# ds4 6289c516 --batched-session 16 vs the same exact-token harness at c=1/4/8/16, 1000-in / 1000-out.
set -u
D=/Users/seangherardi/Code/slimserve/SlimServe-glm53f/perf/results/2026-09-24/ds4-batched
WT=/Users/seangherardi/Code/slimserve/SlimServe-glm53f
PY=/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python
GGUF=$HOME/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf
cd ~/.local/scratch/ds4-upstream
python3 -c "import subprocess,os; subprocess.Popen(['./ds4-server','--metal','-m','$GGUF','--ctx','3008','--tokens','2000','--host','127.0.0.1','--port','18080','--power','100','--batched-session',os.environ.get('NSESS','8'),'--mixed-prefill-quantum','1024'], stdout=open('$D/ds4.log','ab'), stderr=subprocess.STDOUT, start_new_session=True)"
for i in $(seq 1 180); do sleep 5; curl -s http://127.0.0.1:18080/v1/models >/dev/null 2>&1 && break; done
NAME=$(curl -s http://127.0.0.1:18080/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "READY name=$NAME"
cd $WT
for c in ${CONC:-1 4 8 16}; do
  echo "[$(date +%T)] c$c"
  PYTHONPATH=$WT $PY benchmarks/benchmark_dsv4_exact.py --model $GGUF --served-model-name "$NAME" --source perf/results/harness_assets/m2_source.txt --url http://127.0.0.1:18080/v1/completions --metrics-url none --allow-no-spec --temperature 0 --warmup-output-tokens 1 --concurrency $c --input-tokens 1000 --output-tokens ${OUT:-1000} --prompt-offset 1 > $D/c$c.json 2> $D/c$c.err
  tail -2 $D/c$c.err
done
$PY $WT/perf/results/2026-09-15/glm53f-q2-conc/summarize.py $D
pkill -f ds4-server
echo "[$(date +%T)] ALL DONE"
