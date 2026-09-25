#!/bin/bash
# Teacher-forced check vs ds4 on a 3000-token prompt (above the 2051 dense
# prefill limit, so prefill runs the sparse-MLA kernel), for the shipped
# MMA kernel (MQA=6) and the per-head kernel (MQA=0), plus the 2500x64 gate.
set -u
WT=/Users/seangherardi/Code/slimserve/SlimServe-glm53f
C=$WT/perf/results/2026-09-15/glm53f-q2-conc
O=$WT/perf/results/2026-09-25/glm53f-q2-oracle-long
PY=/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python
GGUF=$HOME/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf
for M in 6 0; do
  bash $C/stop_server.sh >/dev/null 2>&1
  $PY $C/boot_env.py $O/mqa$M VLLM_QC_MLA_SPARSE_MQA=$M
  for i in $(seq 1 120); do sleep 5
    curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 200 && break; done
  echo "== MQA=$M ready"
  cd $WT && PYTHONPATH=$WT $PY perf/results/2026-09-10/glm53f-q2-oracle/teacher_forced.py --dump $O/ds4_logprobs.json \
    --prompt-file $O/prompt_3000.txt --tokenizer $GGUF --out $O/tf_mqa$M.json 2>&1 | grep -v "INFO\|Warning" | tail -4
  bash $WT/perf/results/2026-09-11/glm53f-q2-baseline/gate.sh $O/mqa$M/gates 2>&1 | grep sha
done
bash $C/stop_server.sh
echo ALL DONE
