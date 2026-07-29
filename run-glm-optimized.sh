#!/usr/bin/env bash
# SlimServe — optimized GLM-5.2-Vision-GGUF server for MI300X.
#
#   ./run-glm-optimized.sh [--tp N] [--quant NAME] [--ctx N] [--port N] [--no-spec]
#
# Defaults target the common case: many short (~2k) requests, with a very large
# ceiling available for the occasional huge one. max_model_len is a ceiling, not
# a per-request reservation — KV is a shared pool, so a big ceiling is free when
# requests are small. The ceiling defaults to 512k on 2 GPUs and 1M on 4+.
set -euo pipefail

MODELS=/home/hotaisle/models
VENV=/home/hotaisle/.venv/bin/python

# Run from a neutral cwd: launching from the repo's parent puts the checkout
# root on sys.path, where `vllm` resolves to a namespace package and fails.
cd /

TP=2
DP=1
EP=0
DRAFT_ARG=
QUANT=Q2_K
CTX=                 # default depends on --tp; see below
PORT=8000
MAX_SEQS=32
SPEC=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp)    TP="$2"; shift 2 ;;
        --dp)    DP="$2"; shift 2 ;;
        --ep)    EP=1; shift ;;
        --draft) DRAFT_ARG="$2"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        --ctx)   CTX="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --max-seqs) MAX_SEQS="$2"; shift 2 ;;
        --no-spec) SPEC=0; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$QUANT" in
    Q6_K)   MODEL="$MODELS/GLM-5.2-Vision-GGUF/UD-Q6_K_XL/GLM-5.2-UD-Q6_K_XL-00001-of-00016.gguf" ;;
    Q4_K)   MODEL="$MODELS/GLM-5.2-Vision-GGUF/antirez-routed/GLM-5.2-UD-Q4_K_RoutedQ4K-00001-of-00010.gguf" ;;
    Q2_K)   MODEL="$MODELS/GLM-5.2-Vision-GGUF/antirez-routed/GLM-5.2-UD-Q2_K_RoutedQ2K-00001-of-00006.gguf" ;;
    IQ2_XXS) MODEL="$MODELS/GLM-5.2-Vision-GGUF/antirez-routed/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K-00001-of-00005.gguf" ;;
    *) echo "unknown quant: $QUANT (Q6_K|Q4_K|Q2_K|IQ2_XXS)" >&2; exit 2 ;;
esac
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 1; }

HF_CONFIG="$MODELS/GLM-5.2-Vision-FP8"     # config.json + tokenizer only

# Speculator. Override with --draft <path-or-hf-repo>. A bare repo id is
# resolved through the HF cache, so the 5.9 GB draft downloads on first run;
# a local checkout is preferred when present.
DEFAULT_DRAFT_REPO=RedHatAI/GLM-5.2-speculator.dspark
DRAFT="${DRAFT_ARG:-}"
if [[ -z "$DRAFT" ]]; then
    if [[ -d "$MODELS/GLM-5.2-speculator.dspark" ]]; then
        DRAFT="$MODELS/GLM-5.2-speculator.dspark"
    else
        DRAFT="$DEFAULT_DRAFT_REPO"
    fi
fi

# KV pool, sized explicitly. Auto-sizing (gpu_memory_utilization) misjudges
# this model at large max_model_len: it over-requests and OOMs at init, and
# being generous instead OOMs at execution because the sparse-MLA and indexer
# top-k workspaces also scale with max_model_len. Budget = free VRAM after
# weights, minus workspace headroom. Measured cost/token: 46.6 KB target,
# +9.2 KB for a turboquant_k8v4 draft.
#
# TP2 holds ~129 GiB of weights+state per GPU, leaving ~63 GiB; TP4/TP8 shard
# the weights so far more is free (the MLA latent itself does NOT shard, so
# extra GPUs buy batch and headroom, not per-token cost).
case "$TP" in
    2) KV_GIB=47 ;;
    4) KV_GIB=90 ;;
    8) KV_GIB=120 ;;
    *) KV_GIB=47 ;;
esac
# Expert parallel needs ~10 GiB more per rank than plain TP at the same GPU
# count (measured: tp4 --ep OOMs at the tp4 budget, 187.6 GiB resident before
# profiling), so give the KV pool back that much.
[[ "$EP" == "1" ]] && KV_GIB=$((KV_GIB - 12))
KV_BYTES=$((KV_GIB * 1024 * 1024 * 1024))

# Default context ceiling by GPU count. On 2 GPUs the weights leave only ~63
# GiB per card, and a 1M ceiling needs ~52 GiB of KV *plus* GiB-scale
# sparse-MLA/indexer workspace that also scales with max_model_len — so 1M on
# TP2 only fits with speculation off. From 4 GPUs up the weights shard and
# there is ample room, so 1M is the default there.
if [[ -z "$CTX" ]]; then
    if (( TP >= 4 )); then CTX=1048576; else CTX=524288; fi
fi
if (( TP < 4 )) && (( CTX > 524288 )) && [[ "$SPEC" == "1" ]]; then
    echo "note: ${CTX}-token ceiling on tp=$TP needs speculation off; " \
         "use --no-spec, --ctx 524288, or --tp 4." >&2
fi

SPEC_CFG=""
if [[ "$SPEC" == "1" ]]; then
    # num_speculative_tokens=3 measured best at batch scale: mean accept length
    # only grows 2.70 -> 3.07 from 3 to 7 draft tokens while verify width
    # doubles, so 7 loses ~23% throughput at batch 64. TurboQuant draft KV
    # (turboquant_k8v4) reaches fp8 acceptance parity by ~4k context and is
    # ~22% smaller per token.
    SPEC_CFG=$(cat <<JSON
{"model": "$DRAFT", "method": "dspark", "num_speculative_tokens": 3,
 "attention_backend": "TURBOQUANT", "kv_cache_dtype": "turboquant_k8v4"}
JSON
)
fi

echo "SlimServe: $QUANT  tp=$TP  dp=$DP  ep=$EP  ctx=$CTX  max_seqs=$MAX_SEQS" \
     "kv=${KV_GIB}GiB  spec=$SPEC"
[[ "$SPEC" == "1" ]] && echo "  draft: $DRAFT"

# AITER is required: the target's sparse-MLA indexer has no non-AITER ROCm path.
export VLLM_ROCM_USE_AITER=1

ARGS=(
    --model "$MODEL"
    --hf-config-path "$HF_CONFIG"
    --tokenizer "$HF_CONFIG"
    --trust-remote-code
    --served-model-name GLM-5.2-Vision
    --tensor-parallel-size "$TP"
    --data-parallel-size "$DP"
    --max-model-len "$CTX"
    --max-num-seqs "$MAX_SEQS"
    --max-num-batched-tokens 16384      # 8192 breaks torch.compile range setup
    --kv-cache-memory-bytes "$KV_BYTES"
    --block-size 64                     # sparse MLA + sparse SWA require a multiple of 64
    --enable-prefix-caching
    --enable-chunked-prefill
    --limit-mm-per-prompt '{"vision_chunk": 1}'
    # REQUIRED. Short prompts otherwise take sparse MLA's dense forward_mha
    # path, which the ROCm sparse backend does not implement — the engine dies
    # with NotImplementedError on the first small request.
    --attention-config '{"sparse_mla_force_mqa": true}'
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    --port "$PORT"
)
[[ -n "$SPEC_CFG" ]] && ARGS+=(--speculative-config "$SPEC_CFG")
# Expert parallel: shard the 256 routed experts across ranks instead of
# splitting every expert's matrices (which is what plain TP does).
[[ "$EP" == "1" ]] && ARGS+=(--enable-expert-parallel)

exec "$VENV" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
