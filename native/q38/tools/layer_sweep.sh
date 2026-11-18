#!/bin/bash
# MLP sensitivity by depth: restore gate_up + down of 8 consecutive layers to
# the reference pack, score on the GPU engine. One group per line of output.
#   layer_sweep.sh BASE.q38w REF.q38w      (EVAL_REF / EVAL_SET / OUT as in sens_sweep.sh)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HOME/Code/slimserve/SlimServe/.venv/bin/python}"
S="$HOME/.local/scratch/qwen38-campaign/ref"
BASE="$1"; REF="$2"; TMP="$HOME/models/q38/_lsens.q38w"
for g in 0 1 2 3 4 5 6 7; do
  mix=""
  for l in $(seq $((g * 8)) $((g * 8 + 7))); do mix="${mix}L$l.gate_up=$REF;L$l.down=$REF;"; done
  echo "=== layers $((g * 8))-$((g * 8 + 7))"
  "$PY" "$here/mixpack.py" --out "$TMP" --mix "${mix}*=$BASE" | tail -1
  "$PY" "$here/kld.py" engine --ref "$S/${EVAL_REF:-ref_eval.npz}" --set "$S/${EVAL_SET:-eval_set.json}" --pack "$TMP" --chunk 4 | tail -1 \
    | sed "s|^{|{\"restored\": \"mlp $((g * 8))-$((g * 8 + 7))\", \"base\": \"$(basename "$BASE")\", |" | tee -a "$S/${OUT:-layer_sens_results.jsonl}"
done
rm -f "$TMP"
