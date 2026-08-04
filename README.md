# SlimServe

By Eric Hartford, QuixiAI

*A simplicity-first, opinionated inference engine for the antirez GGUF quants —
ds4's interface, vLLM's engine, and every performance trick we can land.*

<img width="480" height="480" alt="image" src="https://github.com/user-attachments/assets/cbc419c0-2bb7-4294-be1c-121f1c8121b0" />

### With SlimServe you can run GLM-5.2-Vision at

Aggregate throughput, by concurrent requests:

| | 1 | 4 | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **2× MI300X** | **82** | **141** | **176** | **260** | **297** | **408** |
| **4× MI300X** | **104** | **142** | **260** | **270** | **384** | **475** |
| **8× MI300X** | **111** | **212** | **333** | **501** | **607** | † |
| **4× A100** | **66** | **138** | **204** | **273** | **326** | **392** |
| **8× A100** | **81** | **167** | **305** | **333** | **550** | **626** |

<sub>Aggregate tok/s — total tokens generated divided by the time to drain the
whole batch. Per-request rates and latencies are in
[Performance](#performance). † 8 GPUs at 64 concurrent trips an illegal memory
access in vLLM's spec-decode rejection sampler during startup profiling; the
8× row was measured at `--max-seqs 32`, the 2×/4× rows at 64. A100 rows: same
methodology as the MI300X rows (varied real prompts, temperature 0, natural
stops, aggregate = total tokens / drain time). Q2_K, `max_model_len` 4096,
`max_num_seqs` 64, fp8 KV, DSpark k=3 with the draft replicated per GPU,
CUDA graphs; the A100 serving path is fully native CUDA (verified serving
with the `triton` package absent). An 8-GPU A100 box can also run two
independent TP4 instances for isolation, at roughly the 4× row each.</sub>

### …while supporting up to 1 Million tokens of context

<sub>Measured on Hot Aisle MI300X: Q2_K quant, 100k-token prompts, 2k-token
responses, temperature 0, DSpark speculative decoding with TurboQuant draft
KV. 1M-token context is available on 4+ GPUs.</sub>

---

## ⚡ Made possible by the [QuixiCore Kernel Library](https://github.com/QuixiAI/QuixiCore-rocm)

The throughput above is not vLLM's stock ROCm path. It comes from
**[QuixiCore-rocm](https://github.com/QuixiAI/QuixiCore-rocm)** — hand-tuned
gfx942 kernels for the operations this model actually spends its time in: the
Q2_K format layer, MFMA quantized GEMMs, MoE grouped GEMM, and MLA decode.
Without those kernels a 244 GiB routed-MoE GGUF simply does not serve at these
rates on two GPUs.

---

## What SlimServe is

**A simplicity-first inference engine for the [antirez][antirez] GGUF quants,
with vision enabled wherever the model has it.**

One command, a fixed set of tested profiles, no flag archaeology:

```bash
slimserve                 # pick a profile, then chat
slimserve glm52-2         # or name one
slimserve k3-6 --serve    # OpenAI-compatible endpoint
```

It is **opinionated**. Every configuration it will run lives in
[`slimserve/profiles.json`](slimserve/profiles.json), and it refuses anything
that is not in there rather than letting you discover the hard way, eight hours
into a 244 GiB download, that the combination does not fit. Settings are the
measured ones; the file records why each number is what it is.

It is **inspired by [ds4][ds4]** — antirez's from-scratch C engine — and takes
its interface philosophy from it: the model's voice on stdout and the tool's on
stderr, Ctrl-C stops generation without leaving the prompt, one line of rates
after each turn. Where ds4 is built from scratch against llama.cpp's world,
SlimServe is built from vLLM, so it inherits continuous batching, paged KV,
prefix caching and a production HTTP server.

It is **performance-focused, and every bleeding-edge trick is in scope.**
Right now that means DSpark speculative decoding against a GGUF-quantized
verifier and TurboQuant compressed KV for the draft. That list is expected to
turn over as better tricks appear — this is not a codebase that will refuse a
hack because it is exotic.

### Hardware

| Runs today | Coming |
| --- | --- |
| AMD MI300X (2–8 GPUs) | RTX 3090 / 4090 / 5090 |
| NVIDIA A100 (4–8 GPUs) | RTX PRO 6000 |
| Apple Metal | NVIDIA DGX Spark, multi-node |

### Models

Three architectures: [GLM-5.2-Vision][glm] and [Kimi K3](#also-supported-kimi-k3-vision)
with vision, [DeepSeek-V4-Flash](#also-supported-deepseek-v4-flash-text-only)
text-only. GLM-5.2-Vision is what the tuning targets; the others reuse its
kernels. The [profile table](#quick-start) below has the GPU counts.

This is not general-purpose vLLM. Support for models, accelerators and
quantization paths outside the ones above has been deleted so the rest can be
tuned hard. If you want to serve something else, use
[upstream vLLM](https://github.com/vllm-project/vllm).

What the specialization buys:

- [QuixiCore-rocm](https://github.com/QuixiAI/QuixiCore-rocm) HIP/MFMA kernels
  for the routed Q2_K/Q4_K MoE and dense projections
- DSpark speculative decoding against a GGUF-quantized verifier
- TurboQuant compressed KV for the draft model (sliding-window support added here)
- AITER sparse-MLA (DSA) attention with a working 1M-token path
- ~154 s cold start on a 244 GiB model (even faster load time than llama.cpp)

[antirez]: https://huggingface.co/antirez
[ds4]: https://github.com/antirez/ds4
[glm]: https://huggingface.co/QuixiAI/GLM-5.2-Vision-GGUF

---

## Quick start

```bash
slimserve                      # pick a profile, then chat
slimserve glm52-2              # chat on 2 GPUs
slimserve glm52-4 --serve      # OpenAI-compatible endpoint on :8000
slimserve k3-6 -p "2 + 2?"     # one shot, then exit
```

`slimserve` runs a fixed set of tested configurations and refuses everything
else. Weights download on first use into `~/models` (override with `--cache` or
`$SLIMSERVE_CACHE`).

Output streams token by token in both modes. The prompt is an SSE client of the
same OpenAI-compatible endpoint `--serve` exposes, so an interactive answer and
an API answer come from one engine with one configuration.

| Profile | Model | GPUs | Runs on |
| --- | --- | ---: | --- |
| `glm52-2` | GLM-5.2-Vision | 2 | MI300X |
| `glm52-4` | GLM-5.2-Vision | 4 | MI300X, A100 |
| `glm52-8` | GLM-5.2-Vision | 8 | MI300X, A100 |
| `dsv4-2` | DeepSeek-V4-Flash (text) | 2 | MI300X |
| `dsv4-4` | DeepSeek-V4-Flash (text) | 4 | MI300X |
| `k3-6` | Kimi K3 | 6 | MI300X |
| `k3-8` | Kimi K3 | 8 | MI300X |
| `k3-8t` | Kimi K3 | 8 | MI300X |

GLM takes `--quant IQ2_XXS|Q2_K|Q4_K` (Q4_K needs 4+ GPUs). DeepSeek-V4-Flash
takes `--quant MXFP4|Q4_K|Q4K-tail|IQ2_XXS`, the four 0731 builds; the two
larger ones need 4 GPUs. Kimi K3 has one published quant. `slimserve --list`
shows every profile and why any of them will not run here;
`slimserve <profile> --dry-run` prints the resolved settings without loading
anything.

**Every legal configuration lives in
[`slimserve/profiles.json`](slimserve/profiles.json)** — model sources, download
URLs, per-platform engine settings, and the minimum GPU count for each quant.
That file is the authority; nothing outside it can be run.

The lower-level `run-glm-optimized.sh` still exists for GLM experiments that
step outside the profiles (arbitrary `--tp`, `--ctx`, `--kv`, `--ep`).

---

## Getting the weights manually

`slimserve` fetches what a profile needs, so this section is only for setting up
a cache by hand or for the lower-level script. The GGUF repo is 845 GB in
total, so fetch only the quant you intend to serve plus the shared vision
projector.

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

## Running the lower-level script

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

**The split depends on the reasoning parser, and its default is GLM's.** This
fork defaults `reasoning_parser` to `glm45`, which is right for GLM-5.2 and
wrong for everything else: pointed at another model's output it classifies the
*whole* reply as reasoning, so `content` is `null` however large `max_tokens`
is. That looks exactly like a truncated think but never resolves. The
`slimserve` profiles pin the correct parser per model; if you launch the API
server by hand, pass `--reasoning-parser` to match the model you loaded.

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

## Also supported: DeepSeek-V4-Flash (text only)

The second architecture this fork serves. Text only — the model has no vision
tower, so none of the mmproj path applies.

```bash
slimserve dsv4-4                    # MXFP4 on 4 GPUs, the tuned path
slimserve dsv4-2 --quant IQ2_XXS    # smallest, 2 GPUs
slimserve dsv4-4 --serve            # OpenAI-compatible endpoint
```

All four 0731 quants from [antirez/deepseek-v4-gguf][ds4w] load and serve.
They differ only in how the routed experts are stored — every other tensor is
the same F32/F16/Q8_0 in all of them — so the adapter is quant-agnostic and the
choice is purely quality against footprint.

| Expert quant | Size | Min GPUs | Notes |
| --- | ---: | ---: | --- |
| `MXFP4Experts` | 145 GiB | 4 | Tuned path; own MXFP4 HIP kernels. |
| `Q4KExperts` | 153 GiB | 4 | Highest quality of the four; imatrix. |
| `Layers37-42Q4KExperts` | 91 GiB | 2 | Mixed; see note below. |
| `IQ2XXS-w2Q2K` | 81 GiB | 2 | Smallest; IQ2_XXS experts, Q2_K down. |

Drafter (all quants): DSpark MXFP4-Q8_0 — [alessandrobologna][ds4d].
Context 1048576 (yarn, 65536 base).

`Layers37-42Q4KExperts` puts Q4_K on the last six expert layers and
IQ2_XXS gate/up with Q2_K down on the rest, which is why it is the one file
carrying three expert quants at once.

IQ2_XXS has no MMQ tile kernel, so the two quants that use it stay on the
vector MoE path at every batch size. That is a throughput ceiling, not a
correctness limit.

[ds4w]: https://huggingface.co/antirez/deepseek-v4-gguf
[ds4d]: <https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF> <!-- markdownlint-disable-line MD013 -->

The equivalent by hand, which is what `slimserve dsv4-4` runs:

```bash
GGUF=$MODELS/antirez-deepseek-v4-gguf
VLLM_ROCM_USE_AITER=1 python -m vllm.entrypoints.openai.api_server \
  --model $GGUF/DeepSeek-V4-Flash-...-mxfp4-0731.gguf \
  --trust-remote-code --served-model-name DeepSeek-V4-Flash \
  --tensor-parallel-size 4 --block-size 256 \
  --reasoning-parser deepseek_v4 \
  --attention-config '{"sparse_mla_force_mqa": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

Three flags are not optional. `--block-size 256` is what the DeepSeek-V4 sparse
MLA backend supports; the GLM value of 64 fails at KV-cache setup with "no
common block size". `sparse_mla_force_mqa` is required for the same reason it
is on the GLM path — short prompts otherwise take a dense forward the ROCm
sparse backend does not implement. `--reasoning-parser deepseek_v4` overrides
the GLM default, which would otherwise return every answer as `reasoning` with
`content: null` — see [Reasoning output](#reasoning-output).

Everything is read from the GGUF: no `--hf-config-path`, no `--tokenizer`. The
routed experts are MXFP4, which this fork reads through its own HIP dequant,
GEMV and MMQ tile kernels, and the DSA indexer (64 heads, 128 key length) runs
on the bitwise-verified MFMA `fp8_mqa_logits`.

---

## Also supported: Kimi K3 (vision)

The third architecture, and the largest thing this fork serves: an 858 GB
IQ2_XXS/Q2_K GGUF with 896 routed experts over 93 layers (69 KDA + 24 MLA), plus
a BF16 vision tower.

```bash
slimserve k3-8t           # tensor parallel across all 8 GPUs — fastest
slimserve k3-6            # tensor parallel across 6 GPUs
slimserve k3-8            # 4 replicas x TP2, experts split 112-per-rank
```

| Profile | Topology | Why |
| --- | --- | --- |
| `k3-8t` | TP8 | 12 attention heads per rank, decoded by the HIP MLA kernel. |
| `k3-6` | TP6 | 16 heads per rank; leaves two cards idle. |
| `k3-8` | DP4 × TP2, EP8 | Aggregate throughput at high concurrency. |

**On TP8.** 96 attention heads over 8 ranks gives 12 per rank, which AITER's MLA
cannot run: its gfx942 decode ships as pre-assembled code objects with the query
head count baked in, so only multiples and divisors of 16 work. That made TP8
unservable, and `k3-8` existed to reclaim the two cards TP6 leaves idle by
running four TP2 replicas instead. But four two-GPU replicas is the wrong shape
for latency — one replica serves any single request. `csrc/quixicore/rocm/`
now carries a HIP decode kernel that takes the head count as a grid dimension,
so TP8 runs, and `k3-8t` puts all eight cards on every request.

Aggregate tok/s, 1k in / 2k out, `--ignore-eos`:

| Profile | 1 | 4 | 8 |
| --- | --- | --- | --- |
| `k3-8t` (TP8, HIP MLA) | **34.6** | **106.4** | **149.8** |
| `k3-6` (TP6, AITER MLA) | 32.8 | 91.7 | 122.5 |
| `k3-8` (DP4 × TP2) | 11.3 | — | — |

Two things are required and are set by the profiles: `kv_cache_dtype=auto`,
because this fork defaults the cache to fp8 for GLM and K3 cannot use it, and
`HSA_XNACK=1` for the load. The text GGUF ships as five parts to concatenate,
and its published header has `kimi-k3.vision = false`, which is wrong; the
downloader reassembles the parts and corrects that byte. The vision projector
comes from `unsloth/Kimi-K3-GGUF`, not from the text repo.

Both profiles need a `_C_stable_libtorch` built from this tree, `k3-6` most of
all. The MMQ expert-id fix lives in a `.cuh`, and a stale extension silently
zeroes every expert above 255 — 71% of this model's experts — in any
prefill-width MoE call that sees global expert ids, which is exactly what
tensor parallel does. Under `k3-8` each rank holds ids 0–111 and never reaches
the ceiling, so `k3-6` is the configuration a stale build degrades.

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

**More GPUs: scale with `--tp`, not `--dp`**

```bash
./run-glm-optimized.sh --tp 4     # 1.2x single-request, 1.3x at concurrency 16 vs TP2
./run-glm-optimized.sh --tp 8     # Q6_K territory; also the most KV headroom
```

See [Parallelism: TP vs DP vs EP](#parallelism-tp-vs-dp-vs-ep) for why
`--dp` is usually the wrong knob.

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

**2× MI300X**

| Concurrent requests | Aggregate tok/s | Per-request tok/s | Median latency |
| ---: | ---: | ---: | ---: |
| 1 | 81.6 | 81.6 | 25 s |
| 4 | 141.3 | 38.9 | 51 s |
| 8 | 176.2 | 23.6 | 83 s |
| 16 | 260.2 | 17.7 | 113 s |
| 32 | 296.9 | 10.0 | 204 s |
| 64 | 407.9 | 7.0 | 288 s |

**4× MI300X**

| Concurrent requests | Aggregate tok/s | Per-request tok/s | Median latency |
| ---: | ---: | ---: | ---: |
| 1 | 104.3 | 104.3 | 19 s |
| 4 | 141.8 | 41.6 | 48 s |
| 8 | 260.4 | 36.2 | 56 s |
| 16 | 270.0 | 24.9 | 80 s |
| 32 | 384.4 | 15.3 | 132 s |
| 64 | 474.8 | 10.5 | 188 s |

**8× MI300X** (`--max-seqs 32`; see the footnote on the headline table)

| Concurrent requests | Aggregate tok/s | Per-request tok/s | Median latency |
| ---: | ---: | ---: | ---: |
| 1 | 110.7 | 110.7 | 18 s |
| 4 | 212.3 | 56.1 | 38 s |
| 8 | 333.1 | 45.1 | 45 s |
| 16 | 501.4 | 33.5 | 61 s |
| 32 | 606.5 | 20.3 | 100 s |

How to read this: on 2 GPUs **aggregate throughput scales 5× from 1 to 64
concurrent requests, while each individual request gets 12× slower.** Decode at
100k context is memory-bandwidth bound — every step re-reads a large KV working
set — so extra concurrency mostly contends rather than filling idle compute.

Pick a concurrency limit from your latency target, not from peak throughput:

- **Interactive / chat** (a user waiting on the answer): 1–4. A single request
  runs at 82 tok/s on 2 GPUs (104 on 4), faster than most people read; at 4
  concurrent you still get ~39 tok/s each.
- **Balanced serving**: 8–16. Concurrency 16 on 2 GPUs gives 64% of the peak
  aggregate at 18 tok/s per request — usually the best overall trade.
- **Batch / offline** (nobody is watching): 32–64. Peak aggregate lives here,
  but per-request rates fall to 7–10 tok/s, so only go there when total
  completion time is all that matters.

Prefill of the 100k prefix took 130 s on 2 GPUs (**771 tok/s**), 83 s on 4
(**1,207 tok/s**) and 64 s on 8 (**1,572 tok/s**), and is paid once thanks to
prefix caching — every later request against the same corpus skips it. Prefill
scales far better with GPU count than decode does, because it is compute bound
while decode is limited by KV bandwidth.

**Where the GPUs actually pay off.** At one request, 8 GPUs are only 36% faster
than 2 (111 vs 82 tok/s) — a single decode stream cannot use them. Under load
the gap widens sharply: at 16 concurrent, 8 GPUs deliver 1.9× the throughput of
2 (501 vs 260), and per-request speed holds up far better (33 vs 18 tok/s).
Scale out for concurrency and prefill, not for single-stream latency.

Caveats: aggregate is total tokens divided by time to drain the whole batch, so
it includes the tail where the last requests finish. Requests are admitted in
waves rather than all at once, which is why wall time runs above median latency
at high concurrency. Individual cells carry a few percent of run-to-run
variance.

### Parallelism: TP vs DP vs EP

Same workload (100k shared prefix, 2k outputs, Q2_K, spec-3), 4 GPUs for the
TP4/DP rows. These four rows come from one separate sweep run at
`--max-seqs 32`, so the absolute values differ slightly from the tables above
(measured at `--max-seqs 64`); what matters here is the comparison *between*
parallelism modes, which is internally consistent:

| Config | GPUs | 1 request | 16 concurrent | Prefill |
| --- | ---: | ---: | ---: | ---: |
| TP2 | 2 | 87.0 tok/s | 283.9 tok/s | 772 tok/s |
| **TP4** | 4 | **104.1 tok/s** | **371.4 tok/s** | **1,138 tok/s** |
| TP2 × DP2 | 4 | 19.2 tok/s | 360.8 tok/s | 967 tok/s |
| TP4 + EP | 4 | OOM at boot | — | — |
| TP2 × DP2 + EP | 4 | OOM at boot | — | — |

**Use tensor parallel. It is the right default at every size here.** Going
from TP2 to TP4 buys 20% on single-request decode, 31% at concurrency 16, and
47% on prefill. Decode scales sublinearly because it is memory-bandwidth bound
and the MLA latent KV is replicated on every rank rather than sharded; prefill
scales much better because it is compute bound.

**Data parallel is a trap below saturation.** DP2×TP2 delivers 19.2 tok/s on a
single request — **5.4× slower than TP4 on the same four GPUs**. vLLM's
data-parallel replicas advance in lockstep, so while one replica serves your
request the other runs dummy batches, and every step pays a synchronization.
DP only becomes reasonable once every replica is busy: at concurrency 16 it
reaches 360.8 tok/s, still slightly behind plain TP4. Choose DP only when you
are permanently saturated and want more independent schedulers; otherwise TP
wins on both latency and throughput.

**Expert parallel does not fit on this hardware.** GLM-5.2 has 256 routed
experts with 8 active per token, which normally makes EP attractive — but with
`--enable-expert-parallel` the resident set reached 187.6 GiB per GPU before
KV profiling even began, and both EP configurations OOM'd. `--ep` exists in the
script and reduces the KV budget by 12 GiB to compensate, but at Q2_K on
MI300X the weights are simply too large for the extra EP overhead. Revisit it
with a smaller quant or more GPUs.

Caveat: the TP2 row was measured with a 512k context ceiling and the TP4/DP
rows with 256k. The ceiling changes workspace reservation, not steady-state
decode throughput, so the comparison holds — but the rows are not
bit-for-bit identical configurations.

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
- The DSpark draft checkpoint, downloaded automatically on first run

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
- **[QuixiCore-rocm](https://github.com/QuixiAI/QuixiCore-rocm)** — the kernel
  library this server stands on. The Q2_K format layer, MFMA quantized GEMMs,
  MoE grouped GEMM and MLA decode kernels are what make the numbers above
  possible.
- **[Hot Aisle](https://hotaisle.xyz)** — for outstanding MI300X servers. Every
  number in this README was measured on their hardware. Bare-metal MI300X that
  actually behaves like the spec sheet, with the ROCm stack in good shape and
  none of the noisy-neighbour surprises that make performance work miserable
  elsewhere — this project would have been far harder anywhere else.

## License

Apache 2.0, inherited from vLLM. See [LICENSE](LICENSE).
