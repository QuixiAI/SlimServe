# SlimServe

**QuixiAI/SlimServe** — a vLLM fork stripped down and specialized to serve
[**QuixiAI/GLM-5.2-Vision-GGUF**](https://huggingface.co/QuixiAI/GLM-5.2-Vision-GGUF)
as efficiently as possible on 2–8 AMD MI300X GPUs.

This is not general-purpose vLLM. Support for other models, other accelerators,
and most non-GGUF quantization paths has been deleted so the remaining code can
be tuned hard for one model on one machine. If you want to serve something else,
use [upstream vLLM](https://github.com/vllm-project/vllm).

What the specialization buys:

- HIP/MFMA GGUF kernels for the routed Q2_K/Q4_K MoE and dense projections
- DSpark speculative decoding against a GGUF-quantized verifier
- TurboQuant compressed KV for the draft model (sliding-window support added here)
- AITER sparse-MLA (DSA) attention with a working 1M-token path
- ~154 s cold start on a 244 GiB model

---

## Getting the weights

**Nothing is downloaded automatically for the target model** — the server is
given a path to a specific `.gguf` shard. The GGUF repo is 845 GB in total, so
fetch only the quant you intend to serve plus the shared vision projector.

```bash
export MODELS=~/models   # where run-glm-optimized.sh looks

# 1. One quant — pick ONE of these --include patterns
hf download QuixiAI/GLM-5.2-Vision-GGUF \
  --include "antirez-routed/GLM-5.2-UD-Q2_K_RoutedQ2K-*" \
  --local-dir $MODELS/GLM-5.2-Vision-GGUF          # 244 GiB, default

#   IQ2_XXS: --include "antirez-routed/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K-*"   # 196 GiB
#   Q4_K:    --include "antirez-routed/GLM-5.2-UD-Q4_K_RoutedQ4K-*"                  # 404 GiB
#   Q6_K:    --include "UD-Q6_K_XL/*"                                                # 638 GiB

# 2. Vision projector + chat template (shared by every quant, ~950 MB)
hf download QuixiAI/GLM-5.2-Vision-GGUF \
  --include "mmproj-GLM-5.2-Vision-f16.gguf" "chat_template.jinja" \
  --local-dir $MODELS/GLM-5.2-Vision-GGUF
```

The `--include` patterns end in `-*` so every shard of the chosen quant is
fetched (a quant is split across 5–16 files and all are required).

**The speculator downloads itself.** `run-glm-optimized.sh` passes the HF repo
id `RedHatAI/GLM-5.2-speculator.dspark` (5.9 GB) unless
`$MODELS/GLM-5.2-speculator.dspark` already exists, so the first run pulls it
into the HF cache. Point somewhere else with `--draft <path-or-repo>`, or turn
speculation off with `--no-spec`.

**You also need `config.json` and the tokenizer.** GGUF does not carry the
multimodal config this model needs, so the server is additionally given
`--hf-config-path` (see `HF_CONFIG` in the script) pointing at a directory with
the GLM-5.2-Vision `config.json`, `configuration_glm5v.py`, `tokenizer.json`,
`tokenizer_config.json`, `preprocessor_config.json` and `generation_config.json`.
Only those small files are used — never the weights in that repo.

## Quick start

```bash
./run-glm-optimized.sh                      # Q2_K, TP2, 512k ctx, 32 concurrent
./run-glm-optimized.sh --tp 4               # 4 GPUs → full 1M context by default
./run-glm-optimized.sh --quant Q4_K --tp 4  # higher quality, 4 GPUs
./run-glm-optimized.sh --quant Q6_K --tp 8  # near-lossless, 8 GPUs
```

**Context ceiling scales with GPU count.** The default is 512k on 2 GPUs and
**1M on 4 or more** — see [Context and GPU count](#context-and-gpu-count).

The server exposes the OpenAI-compatible API on `--port` (default 8000) under
the model name `GLM-5.2-Vision`.

### Vision request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.2-Vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"}},
        {"type": "text", "text": "What is happening in this image? Name the two Pokemon and read the interface text."}
      ]
    }],
    "max_tokens": 2048,
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```

### Reasoning output

GLM-5.2 is a reasoning model and thinking is on by default — that is the point
of the model, and you should normally leave it on. This build parses the chain
of thought into a separate `reasoning` field on the message:

```bash
# the model's thinking
... | jq -r '.choices[0].message.reasoning'
# the final answer
... | jq -r '.choices[0].message.content'
```

**Budget `max_tokens` for thinking plus answer.** `max_tokens` covers both, and
`content` stays `null` until the reasoning finishes. If you set it too low the
response comes back with `"content": null` and `finish_reason: "length"` — that
is a truncated think, not an error. A few hundred tokens is rarely enough for a
vision or analysis request; 2048 is a safe starting point, and reasoning-heavy
prompts want more.

If you have a genuinely latency-critical path where you want the answer only,
you can pass `"chat_template_kwargs": {"enable_thinking": false}` per request —
but expect lower quality on anything requiring multi-step reasoning, and prefer
raising `max_tokens` instead.

### Text request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "GLM-5.2-Vision",
       "messages": [{"role": "user", "content": "Explain speculative decoding."}],
       "max_tokens": 512, "temperature": 0}' | jq -r '.choices[0].message.content'
```

---

## Choosing a quant

All routed quants use the *antirez-routed* layout, which keeps the routed-expert
tensors at the named precision while giving attention and shared experts more
bits. Sizes are on-disk totals; you need enough **aggregate** VRAM across your
GPUs for the weights **plus** the KV pool (192 GB per MI300X).

| Quant | Size | Min GPUs | Use when |
| --- | ---: | ---: | --- |
| [`UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K`](https://huggingface.co/QuixiAI/GLM-5.2-Vision-GGUF/blob/main/antirez-routed/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K-00001-of-00005.gguf) | 196 GiB | 2 | Maximum throughput / longest context on 2 GPUs. Noticeably weaker on reasoning and precise transcription; fine for classification, routing, short summarization. |
| [`UD-Q2_K_RoutedQ2K`](https://huggingface.co/QuixiAI/GLM-5.2-Vision-GGUF/blob/main/antirez-routed/GLM-5.2-UD-Q2_K_RoutedQ2K-00001-of-00006.gguf) | 244 GiB | 2 | **Default.** Best quality that still fits 2 GPUs with room for a large KV pool. The configuration this fork is tuned against. |
| [`UD-Q4_K_RoutedQ4K`](https://huggingface.co/QuixiAI/GLM-5.2-Vision-GGUF/blob/main/antirez-routed/GLM-5.2-UD-Q4_K_RoutedQ4K-00001-of-00010.gguf) | 404 GiB | 4 | Quality matters more than GPU count: agentic/tool use, code, math, careful OCR. The best quality/cost point if you have 4 GPUs. |
| [`UD-Q6_K_XL`](https://huggingface.co/QuixiAI/GLM-5.2-Vision-GGUF/blob/main/UD-Q6_K_XL/GLM-5.2-UD-Q6_K_XL-00001-of-00016.gguf) | 638 GiB | 8 | Near-lossless reference. Use for evaluation baselines, or when an accuracy regression must not come from quantization. Lowest throughput per GPU. |

Rules of thumb:

- **Start at Q2_K.** It is the tuned path, and 2 GPUs leaves 6 free for other work.
- **Go up (Q4_K) when the task is precision-sensitive** — multi-step tool calls,
  code generation, math, or reading small text out of images. Quantization error
  compounds across reasoning steps in a way it does not in summarization.
- **Go down (IQ2_XXS) only when throughput or context is the binding constraint**
  and the task is tolerant. Test it on your own eval before trusting it.
- **Q6_K is a measuring stick, not a serving target** on this hardware.
- Speculative decoding costs quality **nothing** — the verifier accepts or
  rejects every draft token — so leave it on regardless of quant.

---

## Configuration

`run-glm-optimized.sh` flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--tp N` | 2 | Tensor-parallel size. Must divide into your GPU count. |
| `--quant NAME` | `Q2_K` | `IQ2_XXS`, `Q2_K`, `Q4_K`, `Q6_K` |
| `--ctx N` | 524288 (TP2) / 1048576 (TP≥4) | Max context ceiling; model max is 1048576. |
| `--max-seqs N` | 32 | Concurrent requests. |
| `--port N` | 8000 | |
| `--draft P` | `RedHatAI/GLM-5.2-speculator.dspark` | Speculator path or HF repo id. |
| `--no-spec` | off | Disable DSpark speculative decoding. |
| `--dp N` | 1 | Data-parallel replicas. |
| `--ep` | off | Shard the 256 routed experts across ranks. |

### Context and GPU count

**On 4 or more GPUs, use the full 1M context.** The weights shard across ranks,
so each card has far more room left for KV — 1M with speculative decoding still
on fits comfortably, and the script defaults `--ctx` to 1048576 at `--tp 4` and
above. Nothing needs to be turned off.

**On 2 GPUs, 512k is the practical ceiling.** The weights leave only ~63 GiB per
card. A 1M ceiling needs ~52 GiB of KV once the draft model's KV is included,
and the sparse-MLA and indexer workspaces — which also scale with
`max_model_len` — need several GiB on top of that. 1M on 2 GPUs therefore
requires giving up speculative decoding:

```bash
./run-glm-optimized.sh --tp 2 --ctx 1048576 --max-seqs 1 --no-spec
```

That configuration is verified working (1,000,021 tokens prefilled and answered
correctly), but it serves one request at a time and prefills at ~541 tok/s —
**~31 minutes for a cold 1M prompt**. If you have 4+ GPUs, prefer those; if you
have 2, prefer 512k and keep speculation.

One caveat that applies at every size: the MLA latent KV does **not** shard with
tensor parallelism — every rank keeps the full latent. Extra GPUs buy you
headroom and batch, not a lower per-token KV cost.

### Why these defaults

**`max_model_len` is a ceiling, not a reservation.** KV is a shared pool, so a
512k ceiling costs nothing when your traffic is 2k-token requests — 32
concurrent 2k requests use ~2 GiB of a 47 GiB pool. Set the ceiling to the
largest request you ever want to accept.

**The KV pool is sized explicitly, not by `gpu-memory-utilization`.** vLLM's
auto-sizer misjudges this model at large `max_model_len`: it over-requests
(64 GiB against 61.85 GiB free → OOM at init), and simply giving it *less* is
not enough either — the sparse-MLA and indexer top-k workspaces also scale with
`max_model_len`, so a 55 GiB pool allocates fine and then OOMs mid-execution
with 8 MiB free. The script sets `--kv-cache-memory-bytes` per TP size with
workspace headroom left over.

**Speculative decoding uses 3 draft tokens, not 7.** Measured at batch 64 on
100k-token contexts: mean acceptance length only grows 2.70 → 3.07 going from 3
to 7 draft tokens, while verification width doubles. Spec-3 gains +12%
throughput; spec-7 *loses* 23%. Use 7 only for single-request latency, where
verify width is nearly free.

**Draft KV uses TurboQuant** (`turboquant_k8v4`), ~22% smaller per token than
fp8 and at acceptance parity by ~4k context.

**`--block-size 64`** is required: the target's sparse MLA and sparse SWA groups
only accept multiples of 64.

**`--max-num-batched-tokens 16384`** — 8192 trips a torch.compile range-setup
assertion at this configuration.

**`VLLM_ROCM_USE_AITER=1`** is set by the script and is mandatory: the sparse
attention indexer has no non-AITER ROCm path.

**`--attention-config '{"sparse_mla_force_mqa": true}'` is mandatory too**, and
its absence is not obvious from long-context testing. Short prompts otherwise
route sparse MLA through its dense `forward_mha` path, which the ROCm sparse
backend does not implement — the engine dies with `NotImplementedError` on the
first small request while 100k-token requests keep working fine.

---

## Example configurations

**Default: many short requests, high concurrency (2 GPUs)**

```bash
./run-glm-optimized.sh --quant Q2_K --tp 2 --ctx 524288 --max-seqs 32
```

**Full 1M context with speculation kept on (4 GPUs)**

```bash
./run-glm-optimized.sh --quant Q2_K --tp 4 --max-seqs 32
```

`--ctx` defaults to 1048576 at TP≥4. This is the recommended way to have 1M
available: sharded weights leave enough VRAM that the huge ceiling costs you
nothing on ordinary short requests.

**Quality-first agentic serving (4 GPUs)**

```bash
./run-glm-optimized.sh --quant Q4_K --tp 4 --ctx 262144 --max-seqs 64
```

**Maximum context on only 2 GPUs**

```bash
./run-glm-optimized.sh --quant Q2_K --tp 2 --ctx 1048576 --max-seqs 1 --no-spec
```

Speculation must be off and concurrency drops to 1. Prefill dominates at
~541 tok/s — **~31 minutes for a cold 1M prompt**. Prefix caching makes that a
one-time cost for a shared corpus, but this is impractical for cold one-shot
traffic. Use 4 GPUs if 1M matters to you.

**Evaluation baseline (8 GPUs)**

```bash
./run-glm-optimized.sh --quant Q6_K --tp 8 --ctx 131072 --max-seqs 16
```

---

## Performance

### Concurrency scaling — 2×MI300X, Q2_K, 100k in / 2k out

Shared 100k-token prefix (prefix-cached), unique per-request question, exactly
2,000 output tokens each, DSpark spec-3 + TurboQuant draft KV, temperature 0:

| Concurrent requests | Aggregate tok/s | Per-request tok/s | Median latency |
| ---: | ---: | ---: | ---: |
| 1 | 87.0 | 87.0 | 23 s |
| 2 | 112.5 | 56.7 | 36 s |
| 4 | 149.7 | 38.1 | 53 s |
| 8 | 193.4 | 24.8 | 80 s |
| 16 | 283.9 | 18.4 | 110 s |
| 32 | 322.3 | 10.6 | 191 s |

How to read this: **aggregate throughput scales 3.7× from 1 to 32 concurrent
requests, while each individual request gets 8.2× slower.** Decode at 100k
context is memory-bandwidth bound — every step re-reads a large KV working set —
so extra concurrency mostly contends rather than filling idle compute.

Pick a concurrency limit from your latency target, not from peak throughput:

- **Interactive / chat** (a user waiting on the answer): 1–4. At 87 tok/s a
  single request is faster than most people read; at 4 concurrent you still get
  38 tok/s each.
- **Balanced serving**: 8–16. Concurrency 16 captures 88% of peak aggregate
  throughput at 18 tok/s per request — the best overall trade.
- **Batch / offline** (nobody is watching): 32. You gain only 14% aggregate
  over concurrency 16 while nearly doubling latency, so go here only when
  total completion time is all that matters.

Prefill of the 100k prefix took 129.6 s (**772 tok/s**) and is paid once thanks
to prefix caching — every later request against the same corpus skips it.

### Speculative decoding — same hardware, batch 64

| Configuration | Aggregate throughput |
| --- | ---: |
| No speculation | 254 tok/s |
| **DSpark, 3 draft tokens** | **285 tok/s** |
| DSpark, 4 draft tokens | 253 tok/s |
| DSpark, 7 draft tokens | 194 tok/s |

Draft acceptance on natural text runs 25–33% of draft tokens (mean acceptance
length ~2.7–3.3) and is flat from 500 to 12,000 tokens of context.

Prefix caching gives an **11.4× TTFT speedup** on a warm 12k-token prompt
(12.55 s → 1.10 s) and hits at 16-token granularity, so partially-overlapping
prompts still benefit.

### Benchmarking caveat

Do **not** benchmark with synthetic repeated-token prompts (`[1000] * N`). On
such input the model emits degenerate output and speculative acceptance
collapses to ~0%, which looks like a serving bug and is not one. Use natural
text with per-request unique suffixes.

---

## Requirements

- 2–8 AMD MI300X (gfx942), ROCm with AITER
- Python 3.12 via `uv`; all commands go through `.venv/bin/python`
- Model weights and the `mmproj` vision tower under `~/models/GLM-5.2-Vision-GGUF/`
- `config.json` + tokenizer from the FP8 repo (`--hf-config-path`), since GGUF
  alone does not carry the multimodal config
- The DSpark draft checkpoint under `~/models/GLM-5.2-speculator.dspark`

## Building

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt
uv pip install -e . --torch-backend=auto     # C++/HIP changes
VLLM_USE_PRECOMPILED=1 uv pip install -e .   # Python-only changes
```

## Acknowledgements

We stood on the shoulders of giants.

- **Eric Hartford**, **QuixiAI** and **LazarusAI** — for creating
  [QuixiAI/GLM-5.2-Vision-GGUF](https://huggingface.co/QuixiAI/GLM-5.2-Vision-GGUF),
  the model this server exists to serve.
- **Antirez** and **Unsloth** — for the GGUF quantization work. The
  `antirez-routed` layouts and the Unsloth Dynamic (`UD-`) quants are what make
  a model this size fit on two GPUs at all.
- **Baseten** — for the multimodal projector.
- **ibrahima2222** — for the GGUF mmproj.
- **[vLLM](https://github.com/vllm-project/vllm)** — for the engine this fork
  is carved out of. Every good idea in the serving path is theirs; the
  specialization is ours.
- **[zAI](https://huggingface.co/zai-org)** — for GLM-5.2 itself.

## License

Apache 2.0, inherited from vLLM. See [LICENSE](LICENSE).
