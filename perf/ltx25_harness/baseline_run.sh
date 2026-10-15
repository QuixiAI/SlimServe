#!/bin/bash
# Baseline timings for dgrauet/ltx-2-mlx on this box. Usage: run.sh <pack-dir> <tag> <mode> [extra flags]
# mode: distilled | two-stage | dfr ; sizes: small = 768x512x121, full = 1536x1024x121
set -u
PACK=$1; TAG=$2; MODE=$3; SIZE=${4:-small}; shift 4 || true
case $SIZE in small) H=512; W=768;; full) H=1024; W=1536;; esac
SCRATCH=${LTX25_SCRATCH:-$HOME/.local/scratch/ltx25}
OUT=$SCRATCH/baseline; LOG=$OUT/$TAG-$MODE-$SIZE.log
PROMPT="A red fox trotting through a snowy pine forest at dawn, soft golden light, gentle camera dolly forward, birds chirping"
# LTX25_ENV_BIN: the bin/ of the environment with the runner's tools, if any
[ -n "${LTX25_ENV_BIN:-}" ] && export PATH=$LTX25_ENV_BIN:$PATH
cd "$SCRATCH/ltx-2-mlx"
echo "== $TAG $MODE $SIZE $(date +%F_%T)" | tee $LOG
/usr/bin/time -l .venv/bin/ltx-2-mlx generate --$MODE -p "$PROMPT" -m $PACK -H $H -W $W -f 121 --frame-rate 24 -s 42 \
   -o $OUT/$TAG-$MODE-$SIZE.mp4 "$@" 2>&1 | tee -a $LOG
echo "== end $(date +%F_%T)" | tee -a $LOG
grep -E 'real|maximum resident' $LOG | tail -2
