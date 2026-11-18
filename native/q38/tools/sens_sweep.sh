#!/bin/bash
# Per-class sensitivity (EVAL_REF / EVAL_SET / OUT select the eval set and result file):
# Per-class sensitivity: start from a base pack, restore ONE projection class
# to the reference pack, score on the GPU engine. Appends to sens_results.jsonl.
#   sens_sweep.sh BASE.q38w REF.q38w
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HOME/Code/slimserve/SlimServe/.venv/bin/python}"
S="$HOME/.local/scratch/qwen38-campaign/ref"
BASE="$1"; REF="$2"; TMP="$HOME/models/q38/_sens.q38w"
for cls in 'L*.qkvz' 'L*.gdn_out' 'L*.qkv' 'L*.o' 'L*.gate_up' 'L*.down' 'lm_head'; do
  echo "=== $cls"
  "$PY" "$here/mixpack.py" --out "$TMP" --mix "$cls=$REF;*=$BASE" | tail -1
  "$PY" "$here/kld.py" engine --ref "$S/${EVAL_REF:-ref_eval.npz}" --set "$S/${EVAL_SET:-eval_set.json}" --pack "$TMP" --chunk 4 | tail -1 \
    | sed "s|^{|{\"restored\": \"$cls\", \"base\": \"$(basename "$BASE")\", \"ref\": \"$(basename "$REF")\", |" | tee -a "$S/${OUT:-sens_results.jsonl}"
done
rm -f "$TMP"
