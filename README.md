# SlimServe

By Eric Hartford, QuixiAI

*A simplicity-first, opinionated inference engine for the antirez GGUF quants —
ds4's interface, vLLM's engine, and every performance trick we can land.*

<!-- markdownlint-disable-next-line MD013 MD033 -->
<img width="480" height="480" alt="SlimServe" src="https://github.com/user-attachments/assets/cbc419c0-2bb7-4294-be1c-121f1c8121b0" />

## With SlimServe you can run GLM-5.2-Vision at

Aggregate throughput, by concurrent requests:

| | 1 | 4 | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **2× MI300X** | **82** | **141** | **176** | **260** | **297** | **408** |
| **4× MI300X** | **104** | **142** | **260** | **270** | **384** | **475** |
| **8× MI300X** | **111** | **212** | **333** | **501** | **607** | † |
| **4× A100** | **66** | **138** | **204** | **273** | **326** | **392** |
| **8× A100** | **81** | **167** | **305** | **333** | **550** | **626** |

<!-- markdownlint-disable MD033 -->
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
KV. 1M-token context is available on 4+ GPUs. Temperature 0 was the method of
record for these historical numbers; current benchmarking uses each model's
shipped sampling defaults, seeded.</sub>
<!-- markdownlint-enable MD033 -->

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
slimserve glm52-q2k-2     # or name one
slimserve k3-xxs-6 --serve  # OpenAI-compatible endpoint
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

Every profile on every platform enables that same pair: DSpark is never
silently dropped, and every draft uses TurboQuant attention with a
`turboquant_k8v4` KV cache. The registry names one blessed Hugging Face download
for each model family.

### Hardware

| Runs today | Coming |
| --- | --- |
| AMD MI300X (2–8 GPUs) | Apple Metal for GLM-5.2-Vision |
| NVIDIA A100 (4–8 GPUs) | RTX 4090 / 5090, RTX PRO 6000 |
| NVIDIA RTX 3090 (8 GPUs, [P2P driver](docs/geforce-p2p.md)) | |
| [Apple Metal](#apple-silicon) (DeepSeek-V4 128 GiB+; Muse-Glimmer 64 GiB+; Qwen3.8 48 GiB+) | NVIDIA DGX Spark, multi-node |

### Models

Six architectures: [GLM-5.2-Vision][glm], [Kimi K3](#also-supported-kimi-k3-vision),
[Muse-Glimmer-30B](#also-supported-muse-glimmer-30b-vision-apple-silicon),
[Qwen3.8-27B](#also-supported-qwen38-27b-vision-apple-silicon), and
[Qwen3.8-Flash-Next](#also-supported-qwen38-flash-next-rtx-3090) with vision,
[DeepSeek-V4-Flash](#also-supported-deepseek-v4-flash-text-only) text-only.
GLM-5.2-Vision is what the ROCm/CUDA tuning targets, and DeepSeek-V4 and
Kimi K3 reuse its kernels; Muse-Glimmer and Qwen3.8-27B are served by the
Metal stack with their own kernel work (fused gated-DeltaNet step, DFlash 2
drafter kernels, hybrid cache pool); Qwen3.8-Flash-Next is the RTX 3090
target, serving its official FP8 checkpoint through Marlin W8A16
expert-parallel MoE. The [profile table](#quick-start) below has the GPU
counts.

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
- A vendored QuixiCore Metal kernel library (native MPS serving for three
  model families: packed sparse-MLA caches, GGUF i-quant/k-quant GEMV/GEMM,
  fused gated-DeltaNet decode, tensor-ops speculative verify)
- DFlash / DFlash 2 block-diffusion drafters on Metal (path-selector
  drafting with per-step dynamic convolutions)

[antirez]: https://huggingface.co/antirez
[ds4]: https://github.com/antirez/ds4
[glm]: https://huggingface.co/QuixiAI/GLM-5.2-Vision-GGUF

---

## Quick start

```bash
slimserve                      # pick a profile, then chat
slimserve glm52-q2k-2          # chat on 2 GPUs
slimserve glm52-q2k-4 --serve  # OpenAI-compatible endpoint on :8000
slimserve k3-xxs-6 -p "2 + 2?" # one shot, then exit
```

`slimserve` runs a fixed set of tested configurations and refuses everything
else. Weights download on first use into `~/models` (override with `--cache` or
`$SLIMSERVE_CACHE`). When a profile pins an exact DSpark file, that drafter is
included in the same download confirmation, checksum-verified, and reused on
later runs.

Output streams token by token in both modes. The prompt is an SSE client of the
same OpenAI-compatible endpoint `--serve` exposes, so an interactive answer and
an API answer come from one engine with one configuration.

Profile ids follow one scheme everywhere: `<model>-<quant>-<gpus>`. A
profile exists for exactly the platforms it is validated on — if it is
listed for your platform it works there, and it refuses to resolve anywhere
else. Quant tags: `xxs` = IQ2_XXS(-Q2_K), `q4ktail` = Q4K-tail, `mxfp4` =
MXFP4, `q4k` = Q4_K, `q2k` = Q2_K, `kdyn` = K-quant dynamic (per-layer
mixed), `q2kxl` = Unsloth dynamic Q2_K_XL, `fp8` = the official block-FP8
checkpoint.

| Profile | Model | GPUs | Runs on | Draft cache |
| --- | --- | ---: | --- | --- |
| `glm52-q2k-2` | GLM-5.2-Vision | 2 | MI300X | DSpark + TurboQuant |
| `glm52-q2k-4` | GLM-5.2-Vision | 4 | MI300X, A100 | DSpark + TurboQuant |
| `glm52-q2k-8` | GLM-5.2-Vision | 8 | MI300X, A100 | DSpark + TurboQuant |
| `dsv4-xxs-1` | DeepSeek-V4-Flash (text) | 1 | MI300X, Mac | DSpark + TurboQuant |
| `dsv4-q4ktail-2` | DeepSeek-V4-Flash (text) | 2 | MI300X, A100 | DSpark + TurboQuant |
| `dsv4-q4ktail-4` | DeepSeek-V4-Flash (text) | 4 | A100 | DSpark + TurboQuant |
| `dsv4-q4ktail-8` | DeepSeek-V4-Flash (text) | 8 | A100 (TP4 x DP2) | DSpark + TurboQuant |
| `dsv4-mxfp4-4` | DeepSeek-V4-Flash (text) | 4 | MI300X, A100 | DSpark + TurboQuant |
| `dsv4-mxfp4-8` | DeepSeek-V4-Flash (text) | 8 | A100 | DSpark + TurboQuant |
| `dsv4-q4k-8` | DeepSeek-V4-Flash (text) | 8 | MI300X | DSpark + TurboQuant |
| `k3-xxs-6` | Kimi K3 | 6 | MI300X | DSpark + TurboQuant |
| `k3-xxs-8` | Kimi K3 | 8 | MI300X | DSpark + TurboQuant |
| `muse-kdyn-1` | Muse-Glimmer-30B | 1 | Apple Silicon | DFlash |
| `qwen38-q2kxl-1` | Qwen3.8-27B | 1 | Apple Silicon | DFlash 2 |
| `qwen38fn-fp8-8` | Qwen3.8-Flash-Next | 8 | RTX 3090 | MTP (built-in) |
| `glm52-xxs-1` † | GLM-5.2-Vision | 1 | Apple Silicon | DSpark + TurboQuant |

† The GLM Apple Silicon variant is described but not yet runnable. DeepSeek-V4
is measured and supported; see [Apple Silicon](#apple-silicon).

GLM takes `--quant IQ2_XXS|Q2_K|Q4_K` (Q4_K needs 4+ GPUs). DeepSeek-V4-Flash
takes `--quant MXFP4|Q4_K|Q4K-tail|IQ2_XXS`, the four 0731 builds; the two
larger ones need 4 GPUs. Kimi K3 has one published quant. Muse-Glimmer takes
`--quant kquant-dynamic|kquant-17gb`; Qwen3.8-27B has one published quant
(Unsloth dynamic Q2_K_XL). Both are vision models served with their DFlash
block-diffusion drafters. Qwen3.8-Flash-Next serves the official FP8
checkpoint with its own single-layer MTP drafter. `slimserve --list`
shows every profile and why any of them will not run here;
`slimserve <profile> --dry-run` prints the resolved settings without loading
anything.

**Every legal configuration lives in
[`slimserve/profiles.json`](slimserve/profiles.json)** — model sources, download
URLs, per-platform engine settings, and the minimum GPU count for each quant.
That file is the authority; nothing outside it can be run.

The lower-level `run-glm-optimized.sh` still exists for the three exact GLM
quants. Like the profiles, it always enables the matching DSpark draft with a
TurboQuant cache.

### Validate every compatible profile

The live smoke runner discovers every registry profile compatible with the
current machine; it does not keep a separate list that can silently omit a new
TP size. Each resolved plan must use its registered DSpark drafter and
TurboQuant KV cache. It sends text and image requests to GLM and Kimi, and a
text request to DeepSeek:

```bash
source .venv/bin/activate
python benchmarks/smoke_profiles.py
```

The runner attempts the complete matrix even if one profile fails, isolates
each model in its own server process, and prints a JSON result with the log path
for every profile. Use repeated `--profile` options for a focused subset, or
`--download-missing` to fetch registered artifacts non-interactively.

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

#   IQ2_XXS (196 GiB):
#     --include "antirez-routed/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K-*"
#   Q4_K (404 GiB):
#     --include "antirez-routed/GLM-5.2-UD-Q4_K_RoutedQ4K-*"

# 2. Vision projector + chat template (shared by every quant, ~950 MB)
hf download QuixiAI/GLM-5.2-Vision-GGUF \
  --include "mmproj-GLM-5.2-Vision-f16.gguf" "chat_template.jinja" \
  --local-dir $MODELS/GLM-5.2-Vision-GGUF
```

The `--include` patterns end in `-*` so every shard of the chosen quant is
fetched (a quant is split across 5–16 files and all are required).

**The speculator downloads itself.** `run-glm-optimized.sh` passes the exact HF
repo id `RedHatAI/GLM-5.2-speculator.dspark` (5.9 GB) unless
`$MODELS/GLM-5.2-speculator.dspark` already exists, so the first run pulls it
into the HF cache.

## Running the lower-level script

```bash
./run-glm-optimized.sh                      # Q2_K, TP2, 512k ctx, 32 concurrent
./run-glm-optimized.sh --tp 4               # 4 GPUs → full 1M context by default
./run-glm-optimized.sh --quant Q4_K --tp 4  # higher quality, 4 GPUs
```

**Context ceiling scales with GPU count.** The default is 512k on 2 GPUs and
**1M on 4 or more** — see [Context and GPU count](#context-and-gpu-count).

The server exposes the OpenAI-compatible API on `--port` (default 8000) under
the model name `GLM-5.2-Vision`.

### Vision request

<!-- markdownlint-disable MD013 -->
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
    "max_tokens": 2048
  }' | jq -r '.choices[0].message.content'
```
<!-- markdownlint-enable MD013 -->

### Reasoning output

GLM-5.2 is a reasoning model and thinking is on by default — that is the point
of the model, and you should normally leave it on. This is SlimServe policy,
not a GLM special case: every profile on every platform serves with thinking
mode and automatic tool calling enabled by default, with the reasoning and
tool parsers pinned per model in `profiles.json`. This build parses the chain
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
       "max_tokens": 512}' | jq -r '.choices[0].message.content'
```

### Structured output and custom tools

Structured outputs are grammar-enforced through XGrammar: JSON-schema
constrained responses on the OpenAI endpoints, and Responses API custom
tools whose payloads are constrained by Lark grammars — the model cannot
emit an argument that fails the tool's grammar. See
[`examples/responses_apply_patch.py`](examples/responses_apply_patch.py)
for a custom `apply_patch` tool with a Codex-style grammar.

Speculative decoding honors the grammars too: with DSpark, every
sequential draft sample is masked by the request's grammar state, so
constrained requests keep their draft acceptance instead of paying
rejection on every speculative token. Grammar-aware drafting degrades
per-request to unconstrained drafting on any mismatch; target-side
enforcement always stands.

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

Rules of thumb:

- **Start at Q2_K.** It is the tuned path, and 2 GPUs leaves 6 free for other work.
- **Go up (Q4_K) when the task is precision-sensitive** — multi-step tool calls,
  code generation, math, or reading small text out of images. Quantization error
  compounds across reasoning steps in a way it does not in summarization.
- **Go down (IQ2_XXS) only when throughput or context is the binding constraint**
  and the task is tolerant. Test it on your own eval before trusting it.
- Speculative decoding costs quality **nothing** — the verifier accepts or
  rejects every draft token. All profiles use the model family's exact DSpark
  artifact and a `turboquant_k8v4` draft cache.

---

## Also supported: DeepSeek-V4-Flash (text only)

The second architecture this fork serves. Text only — the model has no vision
tower, so none of the mmproj path applies.

```bash
slimserve dsv4-xxs-1                # smallest target on one GPU or a Mac
slimserve dsv4-q4ktail-2            # mixed Q4_K tail on two GPUs
slimserve dsv4-mxfp4-4              # MXFP4 on 4 GPUs, the tuned path
slimserve dsv4-q4k-8                # highest-quality Q4_K on 8 MI300X
slimserve dsv4-mxfp4-4 --serve      # any profile can expose the API
```

All four 0731 quants from [antirez/deepseek-v4-gguf][ds4w] load and serve.
They differ only in how the routed experts are stored — every other tensor is
the same F32/F16/Q8_0 in all of them — so the adapter is quant-agnostic and the
choice is purely quality against footprint.

| Expert quant | Size | Min GPUs (MI300X / A100) | Notes |
| --- | ---: | ---: | --- |
| `MXFP4Experts` | 145 GiB | 4 / 4 | Tuned path; own HIP kernels on MI300X, native CUDA fused/tile/segmented kernels on A100. |
| `Q4KExperts` | 153 GiB | 4 / 4 | Highest quality of the four; imatrix. The MI300X 8-GPU default; serves on A100 via `--quant Q4_K` (generic kernels, unoptimized). |
| `Layers37-42Q4KExperts` | 91 GiB | 1 / 2 | Mixed (`q4ktail`); see note below. The A100 default. |
| `IQ2XXS-w2Q2K` | 81 GiB | 1 / 2 | Smallest; IQ2_XXS experts, Q2_K down. One MI300X or a Mac. |

Artifacts are checksum-pinned: the repo hosts same-size pre-0731 twins of
every 0731 build, so the registry verifies sha256, not just byte counts.

All four use the pinned 6.5 GiB [DeepSeek-V4-Flash 0731 DSpark drafter][ds4d]:
`DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf`. The drafter is
from the same 0731 model revision as the verifier; older DeepSeek-V4-Flash
drafters are not supported. Its sliding-window K/V is stored with TurboQuant.
Context is 1048576 (yarn, 65536 base).

`Layers37-42Q4KExperts` puts Q4_K on the last six expert layers and
IQ2_XXS gate/up with Q2_K down on the rest, which is why it is the one file
carrying three expert quants at once.

On Metal, IQ2_XXS deliberately stays on the device-selected grouped vector MoE
path at every batch size. The padded per-expert tile alternative was slower at
the DeepSeek shapes; grouped dispatch still keeps routing and every selected
expert row on-device, without a per-route host loop.

[ds4w]: https://huggingface.co/antirez/deepseek-v4-gguf
[ds4d]: https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF

The equivalent by hand, which is what `slimserve dsv4-mxfp4-4` runs on
MI300X:

```bash
GGUF=$MODELS/antirez-deepseek-v4-gguf
MODEL=$GGUF/DeepSeek-V4-Flash-...-mxfp4-0731.gguf
DRAFT_DIR=$MODELS/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF
DRAFT=$DRAFT_DIR/DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf
VLLM_ROCM_USE_AITER=1 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --trust-remote-code --served-model-name DeepSeek-V4-Flash \
  --tensor-parallel-size 4 --block-size 256 --max-num-seqs 64 \
  --reasoning-parser deepseek_v4 \
  --attention-config '{"sparse_mla_force_mqa": true}' \
  --compilation-config '{"cudagraph_mode": "NONE"}' \
  --speculative-config "{\"model\": \"$DRAFT\", \"method\": \"dspark\", \
    \"num_speculative_tokens\": 5, \"quantization\": \"gguf\", \
    \"attention_backend\": \"TURBOQUANT\", \
    \"kv_cache_dtype\": \"turboquant_k8v4\"}"
```

The cache and execution settings are deliberate. `--block-size 256` is what
the DeepSeek-V4 sparse MLA backend supports; the GLM value of 64 fails at
KV-cache setup with "no common block size". `sparse_mla_force_mqa` keeps short
prompts off a dense ROCm path that is not implemented. The eager graph mode is
the MI300X form of the stateful DSpark/TurboQuant path, and `max_num_seqs=64`
bounds its startup warmup. `--reasoning-parser deepseek_v4` prevents the GLM
default from returning every answer as reasoning with `content: null`.

On A100 the profiles instead run `FULL_DECODE_ONLY` CUDA graphs with
`max_cudagraph_capture_size: 64`, sized so the concurrency-8 spec-decode
verify step (48 tokens) replays as a graph instead of falling to eager --
worth roughly 2x at concurrency 8. All five A100 profiles
(`dsv4-q4ktail-2/4/8`, `dsv4-mxfp4-4/8`) passed a full 128K-context
lifecycle qualification (1K through 128K cold/hot plus post-128K
continuation, exact token counts at every stage), serve fully native CUDA
(no Triton), and read the routed experts through this tree's own kernels:
the fused per-route decode path at small batches and segmented tensor-core
tiles at prefill widths, for both the MXFP4 and the IQ2_XXS/Q2_K expert
formats.

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
slimserve k3-xxs-6        # best measured single-request latency
slimserve k3-xxs-8        # best measured throughput at concurrency 8
```

| Profile | Topology | Why |
| --- | --- | --- |
| `k3-xxs-6` | TP6 target; replicated draft | Lowest measured c1 latency. |
| `k3-xxs-8` | TP8 target/draft; EP MoE | Highest measured c8 throughput. |

Aggregate tok/s, 1k in / 2k out, `--ignore-eos`, each run gated on the model
answering a known question first:

| Profile | 1 | 8 |
| --- | --- | --- |
| `k3-xxs-6` | **34.0** | **120.4** |
| `k3-xxs-8` | not rerun | **124.6** |

**Why TP8 originally lost to TP6.** TP8 puts 12 attention heads on each rank, which
AITER's MLA cannot run — its gfx942 decode ships as pre-assembled code objects
with the head count baked in. The HIP kernel in `csrc/quixicore/rocm/` takes the
head count as a grid dimension and serves it fine.

The MoE is the real constraint. Tensor-sharding an expert splits its `w2` along
the *packed byte* axis: 1008 bytes of Q2_K over 8 ranks is 126, and the
`type_size` is 84, so every rank would start decoding 1.5 blocks into a
quantized block and the model emits garbage. TP2/4/6 give 6/3/2 whole blocks,
which is why only TP8 breaks. `k3-xxs-8` avoids this with expert parallelism —
each rank holds 112 of the 896 experts whole, so no block is ever cut — but EP
pays an all-to-all dispatch and combine on all 93 MoE layers. The EP-aware,
token-major `w13` path removed the larger non-local expert work and put TP8
slightly ahead at concurrency 8.

Getting tensor-parallel MoE at TP8 would need the intermediate padded from 3072
to 4096 so the byte shard lands on whole blocks, at ~33% more MoE weight memory
with two of eight ranks holding only zeros. Not implemented, and not obviously a
win.

Two things are required and are set by the profiles: `kv_cache_dtype=auto`,
because this fork defaults the cache to fp8 for GLM and K3 cannot use it, and
`HSA_XNACK=1` for the load. The text GGUF ships as five parts to concatenate,
and its published header has `kimi-k3.vision = false`, which is wrong; the
downloader reassembles the parts and corrects that byte. The vision projector
comes from `unsloth/Kimi-K3-GGUF`, not from the text repo.

Both profiles use `Kimi-K3-DSpark-Q8_0.gguf` with a
`turboquant_k8v4` draft cache. At TP6 the draft's 64 query heads, 16 KV heads
and 14336-wide MLP do not divide evenly, so its five-layer backbone and small
Markov head are replicated on each rank while the target and vocabulary logits
remain distributed. TP8 shards the same exact draft normally.

Both profiles need a `_C_stable_libtorch` built from this tree, `k3-xxs-6` most of
all. The MMQ expert-id fix lives in a `.cuh`, and a stale extension silently
zeroes every expert above 255 — 71% of this model's experts — in any
prefill-width MoE call that sees global expert ids, which is exactly what
tensor parallel does. Under `k3-xxs-8` each rank holds ids 0–111 and never reaches
the ceiling, so `k3-xxs-6` is the configuration a stale build degrades.

---

## Also supported: Muse-Glimmer-30B (vision, Apple Silicon)

A 30B vision model served natively on one Mac (64 GiB+) through the
PyTorch-MPS worker and the vendored QuixiCore Metal kernels, with a DFlash
block-diffusion drafter (16 draft tokens per block) verified by fused
tensor-ops kernels.

```bash
slimserve muse-kdyn-1
```

Two published quants from `meta-models/Muse-Glimmer-30B-GGUF`:
`--quant kquant-dynamic` (per-layer mixed, the default) and
`--quant kquant-17gb` (uniform Q4_K_M). Measured 20.1 tok/s single-request
decode on an M5 Max with speculation on (exact-token harness, shipped
sampling defaults, seeded).

## Also supported: Qwen3.8-27B (vision, Apple Silicon)

The smallest-Mac entry point (48 GiB+): a 64-layer hybrid with 48
gated-DeltaNet linear-attention layers and 16 full-attention layers
(GQA 24/4, head_dim 256) over a 248,320-token vocabulary, plus a
qwen3vl-merger vision tower. The linear-attention state and the attention
KV share one hybrid Metal cache pool.

```bash
slimserve qwen38-q2kxl-1
```

One published quant: `unsloth/Qwen3.8-27B-GGUF` UD-Q2_K_XL with its F16
vision projector. The drafter is the Inco AI DFlash 2 block-diffusion model
(`z-lab/Qwen3.8-27B-DFlash2-GGUF`): a path selector scores top-16 candidate
continuations with per-step two-tap dynamic convolutions; the profile runs
3 draft tokens per block. On an M5 Max (exact-token harness, shipped
sampling defaults, seeded): ~17 tok/s plain, 23.1–23.7 tok/s with
speculation on essay-style prompts, and 34.3–35.3 tok/s on GSM8K-style
prompts, where acceptance is high. The `qwen3_5` architecture also loads
the HF safetensors checkpoint (`Qwen/Qwen3.8-27B`) directly, alongside the
GGUF path.

## Also supported: Qwen3.8-Flash-Next (RTX 3090)

`Qwen/Qwen3.8-Flash-Next-FP8` — a 125B-A6B vision-language hybrid (Gated
DeltaNet + Qwen Sparse Attention, 1-in-4 full attention, 4-branch gated
residual) with 51 GiB of n-gram (PLE) embedding tables. This is the 8×
RTX 3090 target:

```bash
slimserve qwen38fn-fp8-8 --serve
```

- **FP8 without FP8 hardware.** SM86 has no FP8 tensor cores, so the experts
  run expert-parallel through Marlin W8A16 block-FP8 kernels (weight-only
  decode to BF16 compute). Expert parallelism is a correctness requirement
  here, not a tuning choice — the block scale geometry doesn't shard under TP.
- **PLE tables in host RAM.** The full 47.7 GiB n-gram table stays pinned in
  host memory per rank; the forward gathers 16 rows per token over UVA, inside
  CUDA graph capture. GPU memory holds weights and KV only.
- **Native 262,144-token context** with bf16 KV, prefix caching, and the
  checkpoint's own single-layer MTP drafter.
- **P2P driver strongly recommended.** Multi-GPU GeForce runs on the stock
  driver but leaves 36–58% of throughput on the table; see
  [docs/geforce-p2p.md](docs/geforce-p2p.md) for QuixiAI's patched
  open-gpu-kernel-modules.

Measured throughput is in [Performance](#8-rtx-3090--qwen38-flash-next-fp8)
below.

## Apple Silicon

Three models run through the in-tree PyTorch-MPS worker and the vendored
QuixiCore Metal kernels. The supported `dsv4-xxs-1` path includes the packed sparse
MLA target cache, hybrid cache allocation, GGUF i-quant/k-quant projections,
the matching 0731 DSpark drafter, and a `turboquant_k8v4` draft cache.
`muse-kdyn-1` (Muse-Glimmer-30B, 64 GiB+) and `qwen38-q2kxl-1` (Qwen3.8-27B,
48 GiB+) serve text and vision with their DFlash block-diffusion drafters —
Qwen3.8's hybrid GDN/attention layers share one Metal cache pool with the
DFlash 2 path-selector drafter. GLM's separate sparse-MLA/vision path remains
individually gated in the CLI.

```console
$ slimserve --list
Profiles (Apple M5 Max, 128 GiB unified):
  ! glm52-xxs-1    GLM-5.2-Vision on one Mac — not validated on Metal
    dsv4-xxs-1     DeepSeek-V4-Flash IQ2_XXS on 1 GPU / Apple Silicon
    muse-kdyn-1    Muse-Glimmer-30B on one Mac
    qwen38-q2kxl-1 Qwen3.8-27B on one Mac
```

**A Mac is gated by memory, not by card count.** Apple exposes one GPU per
machine and weights, KV, and activations share one unified pool. Quants carry
`min_memory_bytes` alongside `min_gpus`, and the registry uses the memory gate
on Metal:

| Mac | DeepSeek-V4-Flash | Muse-Glimmer-30B | Qwen3.8-27B | GLM-5.2-Vision |
| --- | --- | --- | --- | --- |
| 48 GB | — | — | `Q2_K_XL` | — |
| 64 GB | — | both K-quants | `Q2_K_XL` | — |
| 128 GB | `IQ2_XXS`, `Q4K-tail` | both K-quants | `Q2_K_XL` | — |
| 192 GB | + `MXFP4`, `Q4_K` | both K-quants | `Q2_K_XL` | — |
| 256 GB | all four | both K-quants | `Q2_K_XL` | `IQ2_XXS` |
| 512 GB | all four | both K-quants | `Q2_K_XL` | + `Q2_K`, `Q4_K` |

Kimi K3 is deliberately absent: at 800 GiB it does not fit the largest Mac ever
built, so no Metal row claims it.

The figures are physical RAM and include Metal's working-set margin. On the
measured M5 Max, `recommendedMaxWorkingSetSize` is 115.4 of 128 GiB. The default
81 GiB IQ2XXS target plus the 6.5 GiB drafter leaves room for a fixed 1 GiB
hybrid KV pool. The measured pool holds 3,268 cache tokens, enough for one
complete 1k-input/2k-output request under the 3,072-token per-request ceiling.
Higher request counts remain valid API concurrency but drain in scheduler
waves; `max_num_seqs=32` is the admission ceiling, not a claim that 32 full
responses stay resident together.

Metal also has a per-buffer limit below total unified memory. The loader keeps
GGUF tensors as separate MPS allocations rather than trying to place the whole
file in one `MTLBuffer`. The larger 91 GiB Q4K-tail target loads, but the 81 GiB
IQ2XXS build is the measured default because it stays clear of swap once the
drafter and serving caches are resident.

---

## Configuration

`run-glm-optimized.sh` flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--tp N` | 2 | Tensor-parallel size. Must divide into your GPU count. |
| `--quant NAME` | `Q2_K` | `IQ2_XXS`, `Q2_K`, `Q4_K` |
| `--ctx N` | 524288 / 1048576 | TP2 / TP≥4 context ceiling. |
| `--max-seqs N` | 32 | Concurrent requests. |
| `--port N` | 8000 | |
| `--dp N` | 1 | Data-parallel replicas. |
| `--ep` | off | Shard the 256 routed experts across ranks. |

### Context and GPU count

**On 4 or more GPUs, use the full 1M context.** The weights shard across ranks,
so each card has far more room left for KV — 1M with speculative decoding still
on fits comfortably, and the script defaults `--ctx` to 1048576 at `--tp 4` and
above. Nothing needs to be turned off.

**On 2 GPUs, 512k is the ceiling for the required DSpark/TurboQuant path.** The
weights leave only ~63 GiB per card. A 1M ceiling needs ~52 GiB of KV once the
draft cache is included, and the sparse-MLA and indexer workspaces — which also
scale with `max_model_len` — need several GiB on top. Use 4+ GPUs when 1M
context is required.

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

### Default: many short requests, high concurrency (2 GPUs)

```bash
./run-glm-optimized.sh --quant Q2_K --tp 2 --ctx 524288 --max-seqs 32
```

### Full 1M context with speculation kept on (4 GPUs)

```bash
./run-glm-optimized.sh --quant Q2_K --tp 4 --max-seqs 32
```

`--ctx` defaults to 1048576 at TP≥4. This is the recommended way to have 1M
available: sharded weights leave enough VRAM that the huge ceiling costs you
nothing on ordinary short requests.

### More GPUs: scale with `--tp`, not `--dp`

```bash
# 1.2x single-request, 1.3x at concurrency 16 vs TP2
./run-glm-optimized.sh --tp 4
# Most KV headroom for the supported quants
./run-glm-optimized.sh --tp 8
```

See [Parallelism: TP vs DP vs EP](#parallelism-tp-vs-dp-vs-ep) for why
`--dp` is usually the wrong knob.

### Quality-first agentic serving (4 GPUs)

```bash
./run-glm-optimized.sh --quant Q4_K --tp 4 --ctx 262144 --max-seqs 64
```

---

## Performance

### Concurrency scaling — 2×MI300X, Q2_K, 100k in / 2k out

Shared 100k-token prefix (prefix-cached), unique per-request question, exactly
2,000 output tokens each, DSpark spec-3 + TurboQuant draft KV, temperature 0
(the historical method for this table; current runs use the model's shipped
sampling defaults, seeded):

#### 2× MI300X

| Concurrent requests | Aggregate tok/s | Per-request tok/s | Median latency |
| ---: | ---: | ---: | ---: |
| 1 | 81.6 | 81.6 | 25 s |
| 4 | 141.3 | 38.9 | 51 s |
| 8 | 176.2 | 23.6 | 83 s |
| 16 | 260.2 | 17.7 | 113 s |
| 32 | 296.9 | 10.0 | 204 s |
| 64 | 407.9 | 7.0 | 288 s |

#### 4× MI300X

| Concurrent requests | Aggregate tok/s | Per-request tok/s | Median latency |
| ---: | ---: | ---: | ---: |
| 1 | 104.3 | 104.3 | 19 s |
| 4 | 141.8 | 41.6 | 48 s |
| 8 | 260.4 | 36.2 | 56 s |
| 16 | 270.0 | 24.9 | 80 s |
| 32 | 384.4 | 15.3 | 132 s |
| 64 | 474.8 | 10.5 | 188 s |

#### 8× MI300X

`--max-seqs 32`; see the footnote on the headline table.

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

### Apple Silicon — M5 Max, single request

Exact-token harness, each model's shipped sampling defaults, seeded:

| Model | Plain decode | Speculative |
| --- | ---: | ---: |
| Muse-Glimmer-30B (`muse-kdyn-1`) | — | **20.1** tok/s |
| Qwen3.8-27B (`qwen38-q2kxl-1`), essay prompts | 17.0 | **23.1–23.7** tok/s |
| Qwen3.8-27B, GSM8K-style prompts | — | **34.3–35.3** tok/s |

Speculation is always on in the shipped profiles; the plain row is the
diagnostic reference. DeepSeek-V4 on Metal is under re-validation and its
number is deliberately absent (the historical 33.7 tok/s was measured under
a since-changed profile geometry; see `perf/baseline_status.md`).

### 8× RTX 3090 — Qwen3.8-Flash-Next FP8

`qwen38fn-fp8-8`, deployed configuration: native 262,144-token context, bf16
KV, prefix caching, MTP speculation, QuixiAI P2P driver. Exact-token harness
(1,000 in / 2,000 out per request, shipped sampling defaults, seeded):

| Concurrent requests | Aggregate tok/s | Per-request tok/s | Median latency |
| ---: | ---: | ---: | ---: |
| 1 | 139.7 | 139.7 | 14 s |
| 8 | 590.7 | 81.1 | 25 s |
| 32 | 1,151.2 | 39.8 | 50 s |

The c1 row is the median of four seeded runs; single-stream varies
129.8–157.8 tok/s run-to-run with MTP draft-acceptance luck on the sampled
text.

Concurrency 32 is the measured peak for this profile; `max_num_seqs` is set
there deliberately. Peak aggregate throughput scaled 2.8× over the
optimization campaign (409.7 tok/s at bring-up to 1,151.2 now) while the
context ceiling doubled to the model's native 262K. On the stock NVIDIA driver (no GPU-GPU P2P) the same
profile measures 36–58% lower — install the
[P2P driver](docs/geforce-p2p.md).

### Benchmarking caveat

Do **not** benchmark with synthetic repeated-token prompts (`[1000] * N`). On
such input the model emits degenerate output and speculative acceptance
collapses to ~0%, which looks like a serving bug and is not one. Use natural
text with per-request unique suffixes.

---

## Requirements

One of:

- 2–8 AMD MI300X (gfx942), ROCm with AITER
- 4–8 NVIDIA A100, CUDA
- An Apple Silicon Mac (48 GiB+ unified memory; see
  [Apple Silicon](#apple-silicon) for the per-model gates)

Plus:

- Python 3.12 via `uv`; all commands go through `.venv/bin/python`
- Model weights under `~/models/` (or `$SLIMSERVE_CACHE`) — `slimserve`
  downloads or resumes everything a profile needs, including vision
  projectors and draft checkpoints, on first run

## Building

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt
uv pip install -e . --torch-backend=auto     # C++/HIP changes
VLLM_USE_PRECOMPILED=1 uv pip install -e .   # Python-only changes
```

On Apple Silicon the same `uv pip install -e .` builds the `_quixicore_C`
extension and compiles the QuixiCore Metal kernel library
(`vllm/quixicore_metal.metallib`) with the system Metal toolchain; no ROCm
or CUDA components are required.

The ROCm base image pins the exact tested AITER revision and applies
SlimServe's dependency patches before building its wheel. For a source-tree
AITER installation, apply the same patch set as described in
[`docker/patches/aiter/README.md`](docker/patches/aiter/README.md); an
unpatched AITER build can corrupt graph-buffer registration when a captured
graph contains consecutive custom all-reduces.

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
