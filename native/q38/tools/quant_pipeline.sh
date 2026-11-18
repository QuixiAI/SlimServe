#!/bin/bash
# CPU-only quant evaluation chain. Each stage is skipped when its output exists.
#   quant_pipeline.sh            run everything
# Outputs: packs in ~/models/q38, references and logs in the campaign scratch.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HOME/Code/slimserve/SlimServe/.venv/bin/python}"
M="$HOME/models/q38"
S="$HOME/.local/scratch/qwen38-campaign/ref"
LOG="$S/kld_results.jsonl"
stage() { echo "=== $(date +%H:%M:%S) $*"; }

BF16="$M/qwen38-27b-bf16.q38w"
UNSLOTH="$M/qwen38-27b-nvfp4.q38w"
OURS="$M/qwen38-27b-q38nvfp4.q38w"
OURS_IM="$M/qwen38-27b-q38nvfp4-im.q38w"

if [ ! -f "$BF16" ]; then
  stage "pack BF16"
  "$PY" "$here/pack_hf.py" --src "$HOME/models/Qwen3.8-27B" --out "$BF16" --source-name Qwen/Qwen3.8-27B
fi
if [ ! -f "$S/ref_eval.npz" ]; then
  stage "BF16 reference logits (eval set)"
  "$PY" "$here/kld.py" ref --pack "$BF16" --set "$S/eval_set.json" --out "$S/ref_eval.npz"
fi
if [ ! -f "$S/imatrix.npz" ]; then
  stage "BF16 calibration pass (imatrix)"
  "$PY" "$here/kld.py" ref --pack "$BF16" --set "$S/calib_set.json" --out "$S/ref_calib.npz" \
    --imatrix-out "$S/imatrix.npz"
fi
stage "eval: unsloth NVFP4"
"$PY" "$here/kld.py" eval --ref "$S/ref_eval.npz" --set "$S/eval_set.json" --mix "*=$UNSLOTH" --log "$LOG"
if [ ! -f "$OURS" ]; then
  stage "quantize: all-NVFP4, MSE scale search"
  "$PY" "$here/quantize.py" --src "$BF16" --out "$OURS" --jobs 16
fi
stage "eval: all-NVFP4 (ours, plain)"
"$PY" "$here/kld.py" eval --ref "$S/ref_eval.npz" --set "$S/eval_set.json" --mix "*=$OURS" --log "$LOG"
if [ ! -f "$OURS_IM" ]; then
  stage "quantize: all-NVFP4, imatrix-weighted"
  "$PY" "$here/quantize.py" --src "$BF16" --out "$OURS_IM" --jobs 16 --imatrix "$S/imatrix.npz"
fi
stage "eval: all-NVFP4 (ours, imatrix)"
"$PY" "$here/kld.py" eval --ref "$S/ref_eval.npz" --set "$S/eval_set.json" --mix "*=$OURS_IM" --log "$LOG"
stage "done"
