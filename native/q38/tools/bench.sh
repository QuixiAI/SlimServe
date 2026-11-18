#!/bin/bash
# Decode benchmark over the campaign's prompt set at the model's default
# sampling (temperature 1.0, top-k 20, top-p 0.95, seed 42, thinking on).
#   bench.sh PACK [MTP_PACK] [-- extra q38_lab args]
#   DSPINE=drafter.q38w bench.sh PACK -- --depth 4 --adaptive 0.3   (DSpine block drafter)
# Prints per-prompt and aggregate tokens/s (aggregate = total tokens / total
# decode seconds). Set OUT=dir to keep the raw logs.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
B="${BENCH_DIR:-$HOME/.local/scratch/qwen38-campaign/ref/bench}"
GEN="${GEN:-600}"
PACK="$1"; shift
MTP=""; if [ $# -gt 0 ] && [ "$1" != "--" ]; then MTP="$1"; shift; fi
[ "${1:-}" = "--" ] && shift
OUT="${OUT:-$(mktemp -d)}"; mkdir -p "$OUT"
BD="${Q38_BUILD:-$here/../build}"  # engine binaries (a frozen copy while the tree is being changed)
tot_tok=0; tot_s=0
for p in essay code math explain; do
  args=(--pack "$PACK" --metallib "$BD/q38_kernels.metallib:$BD/q38_native.metallib"
        --tokens "$B/$p.tokens" --chunk 8 --gen "$GEN" --seed 42 --out-tokens "$OUT/$p.out.tokens")
  if [ -n "${DSPINE:-}" ]; then args+=(--dspine "$DSPINE" --gen-once)
  elif [ -n "$MTP" ]; then args+=(--mtp "$MTP"); else args+=(--sample); fi
  "$BD/q38_lab" "${args[@]}" "$@" > "$OUT/$p.log" 2>&1
  line=$(grep -E "^(spec )?decode" "$OUT/$p.log")
  tok=$(echo "$line" | sed -E 's/.*: ([0-9]+) tokens in ([0-9.]+) s.*/\1/'); s=$(echo "$line" | sed -E 's/.*: ([0-9]+) tokens in ([0-9.]+) s.*/\2/')
  tot_tok=$((tot_tok + tok)); tot_s=$(echo "$tot_s + $s" | bc -l)
  extra=$(grep -E "tokens/cycle" "$OUT/$p.log" | sed -E 's/^ +//; s/  snapshots.*//' || true)
  printf "%-8s %4d tok %7.3f s  %6.2f tok/s  %s\n" "$p" "$tok" "$s" "$(echo "$tok / $s" | bc -l)" "$extra"
done
printf "AGGREGATE %d tokens in %.2f s = %.2f tok/s\n" "$tot_tok" "$tot_s" "$(echo "$tot_tok / $tot_s" | bc -l)"
