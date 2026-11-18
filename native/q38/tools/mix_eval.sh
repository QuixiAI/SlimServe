#!/bin/bash
# Build one mixed pack, score it (KL on eval2 and eval1) and benchmark it
# (plain and MTP depth 2), appending one JSON line to ref/mix_results.jsonl.
#   mix_eval.sh NAME "glob=pack;...;*=base" [keep]
# The pack is written to ~/models/q38/qwen38-27b-NAME.q38w and removed again
# unless a third argument is given. One GPU process at a time.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HOME/Code/slimserve/SlimServe/.venv/bin/python}"
S="$HOME/.local/scratch/qwen38-campaign/ref"
NAME="$1"; MIX="$2"; KEEP="${3:-}"
OUT="$HOME/models/q38/qwen38-27b-$NAME.q38w"
MTP="${MTP:-$HOME/models/q38/qwen38-27b-mtp-nvfp4.q38w}"
gb=$("$PY" "$here/mixpack.py" --out "$OUT" --mix "$MIX" | tail -1 | sed -E 's/.*read per token ([0-9.]+) GB.*/\1/')
k2=$("$PY" "$here/kld.py" engine --ref "$S/ref_eval2.npz" --set "$S/eval2_set.json" --pack "$OUT" --chunk 4 | tail -1)
k1=$("$PY" "$here/kld.py" engine --ref "$S/ref_eval.npz" --set "$S/eval_set.json" --pack "$OUT" --chunk 4 | tail -1)
spec=$("$here/bench.sh" "$OUT" "$MTP" -- --depth 2 --adaptive 0.5)
plain=$("$here/bench.sh" "$OUT")
"$PY" - "$NAME" "$gb" "$k2" "$k1" "$spec" "$plain" "$MIX" <<'PYEOF' | tee -a "$S/mix_results.jsonl"
import json, re, sys
name, gb, k2, k1, spec, plain, mix = sys.argv[1:8]
k2, k1 = json.loads(k2), json.loads(k1)
def tps(t):
    per = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\w+)\s+\d+ tok\s+[\d.]+ s\s+([\d.]+) tok/s", t, re.M)}
    per["aggregate"] = float(re.search(r"= ([\d.]+) tok/s", t.splitlines()[-1]).group(1))
    return per
print(json.dumps({"name": name, "gb_per_token": float(gb), "kl2": round(k2["mean_kl"], 5), "median2": round(k2["median_kl"], 5),
                  "top1_2": round(k2["top1"], 4), "kl1": round(k1["mean_kl"], 5), "top1_1": round(k1["top1"], 4),
                  "spec": tps(spec), "plain": tps(plain), "mix": re.sub(r"[^=;]*/qwen38-27b-", "", mix)}))
PYEOF
[ -n "$KEEP" ] || rm -f "$OUT"
