# Qwen3.8 on four Intel Arc Pro B70 cards

The active recipe is `qwen38-abliterated-b70-4`: OrcaRouter's native block-FP8
Qwen3.8-27B checkpoint on four 32 GiB Intel Arc Pro B70 cards. The registry,
engine/tool correctness changes, XPU kernels, and checked-in chat template are
all part of the deployment. Copying `profiles.json` alone is insufficient.

This recipe has passed the local tool workload and targeted correctness tests
below. It is **not soak-qualified**: subsequent live traffic encountered a
sampling RPC stall after about five minutes, requiring a backend restart. The
cause remains under investigation. The first recovery restart also stalled
rank3 inside Level Zero on a scalar tensor operation. A targeted Xe GT0 reset
restored all four ranks and health; this is a recovery action, not a proven fix. No full external evaluation was run for the
reported attention comparison.

## Resolved active configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `orcarouter/Qwen3.8-27B-Uncensored-FP8` |
| Revision | `0f3cdb83820a8190ffedaef5b29cf4a635e49b4d` |
| Served model ID | `Qwen3.8-27B-Uncensored-FP8` |
| Weights / linear backend | Offline 128×128 block FP8 E4M3 / `xpu` |
| Tensor parallelism | 4 |
| Maximum context / sequences / batched tokens | 262144 / 32 / 16384 |
| GPU memory utilization | 0.90 |
| KV cache / prefix caching | `fp8_e4m3` / enabled |
| Attention backend | `FLASH_ATTN` using installed XPU kernels |
| Speculation | Native MTP, k=3, TP4 draft, local argmax reduction |
| Graphs | `FULL_DECODE_ONLY`, breakable XPU graphs, compilation mode 0 |
| Graph capture sizes | 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128 |
| Graph replay order | `trail` |
| Triton attention | Tensor descriptors disabled; 16 parallel softmax segments |
| Reasoning / tools | `qwen3` / `qwen3_xml`, automatic tool choice enabled |
| Chat template | `slimserve/chat_templates/qwen38_tool_calling.jinja` |
| Thinking | Enabled, default effort `low`, default budget 2,000 tokens |

The checkpoint already contains FP8 weights; do not add `--quantization fp8`
for online BF16 conversion. Excluded modules remain BF16. The checkpoint has no
calibrated KV scales, so the FP8 KV path uses 1.0. The context limit does not
promise 32 simultaneous full-context requests; available KV capacity governs
admission.

The profile owns `VLLM_TRITON_USE_TD=0`,
`VLLM_XPU_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS=16`, and
`VLLM_XPU_GRAPH_REPLAY_ORDER=trail`. The latter two were previously machine-local
systemd drop-ins and are now recorded in the recipe. The similarly named
`VLLM_TRITON_ATTN_USE_TD` is obsolete and does not select this path. Target
FLASH_ATTN and Triton attention used by the draft can coexist.

The profile's generation defaults set the 2,000-token thinking budget. Explicit
request budgets override that default; `null` or `-1` disables the budget, and
reasoning effort `none` disables thinking. This budget limits reasoning, not the
entire response: leave output-token capacity for the final answer or tool call.
Environment variables inherited by the process override profile environment
values; remove stale experimental overrides when reproducing the recipe.

## Build and dependency requirements

Use the Linux Xe/Level Zero driver, Intel oneAPI DPC++ compiler (`icpx`), and a
PyTorch XPU environment. The measured machine reported these package versions:

| Package | Version |
| --- | --- |
| torch | `2.15.0.dev20260815+xpu` |
| torchvision | `0.30.0.dev20260816+xpu` |
| triton-xpu | `3.8.0+git1e2d42a0` |
| vllm-xpu-kernels | `0.1.13` (locally rebuilt) |
| xgrammar | `0.2.3` |
| intel-sycl-rt | `2026.1.0` |

These are an environment receipt, not a portable binary lockfile. See
`requirements/xpu.txt` and `cmake/xpu.cmake`. Install PyTorch from its XPU index
before building this checkout. Build `vllm-xpu-kernels` against that exact
PyTorch ABI: an older binary loaded successfully but registered operators on
the wrong dispatch key. This dependency supplies FLASH_ATTN and FP8 kernels;
its locally rebuilt binary is not carried by a SlimServe git checkout. Rebuild
or provide a matching artifact before claiming a reproduction. The historical
build needed C++20 and `-Wno-error=deprecated-this-capture`; see the
2026-09-01 entry in `perf/optimization_status.md`.

The native SlimServe XPU target is `_quixicore_C`, with the shared
`quixicore_xpu_ops` library. From an environment with dependencies installed:

```bash
# In the checkout. Keep compiler setup confined to the build subshell.
(
  source /opt/intel/oneapi/setvars.sh
  VLLM_TARGET_DEVICE=xpu .venv/bin/python setup.py build_ext --inplace
)
.venv/bin/python -c 'import vllm._quixicore_C'
```

For an already configured native build directory, use
`cmake --build <build-directory> --target _quixicore_C quixicore_xpu_ops -j4`,
then install/copy both built artifacts into the editable `vllm/` package using
the normal build tooling. The generic CUDA `_C_stable_libtorch` target is not
the XPU build target. Blank `VLLM_XPU_SYCL_TARGETS` uses SPIR-V JIT; changing the
AoT target is optional and was not part of the measured attention A/B.

Do not source oneAPI `setvars.sh` in the serving shell or unit. Its runtime
library path can load a second SYCL runtime beside PyTorch's bundled runtime.
Check that CUDA `triton` has not shadowed `triton-xpu` after dependency changes;
`requirements/xpu.txt` describes the reinstall procedure.

## Start through the profile

Use a checkout with the profile and its dependent code installed into `.venv`.
Adjust the cache location and device selection for the destination machine:

```bash
export SLIMSERVE_CACHE="$HOME/models"
export ZE_AFFINITY_MASK=0,1,2,3
.venv/bin/python -m slimserve.cli qwen38-abliterated-b70-4 --dry-run
.venv/bin/python -m slimserve.cli qwen38-abliterated-b70-4 \
  --serve --host 0.0.0.0 --port 8000 --yes
```

SlimServe checks/downloads the source files into
`$SLIMSERVE_CACHE/Qwen3.8-27B-Uncensored-FP8`. The source requires Hugging Face
access; authenticate on the destination machine when fetching is required.
Existing complete local files can be reused. Credentials and model weights
are not included in this repository.

The accompanying [user service example](qwen38-b70.service.example) assumes
`~/SlimServe` and `~/models`; change those paths if needed. It binds port 8000
directly. Configure access control appropriate to the destination deployment.
Install/start it only when those devices and the port are available:

```bash
mkdir -p "$HOME/.config/systemd/user"
cp docs/deployment/qwen38-b70.service.example \
  "$HOME/.config/systemd/user/slimserve-qwen38-b70.service"
systemctl --user daemon-reload
systemctl --user enable --now slimserve-qwen38-b70.service
curl --fail http://127.0.0.1:8000/health
```

The existing machine instead has a proxy on public **8000** and SlimServe on
**127.0.0.1:8001**. With that topology, replace the service `ExecStart` host/port
with `--host 127.0.0.1 --port 8001`; retain the proxy's existing configuration.
Do not start a second direct-8000 service alongside the proxy. Proxy capture,
credentials, raw request payloads, and external automation are not shipped by
this example.

## Tool behavior and verification

`tool_choice: required` requires at least one call. For functions without
explicit `strict: true`, arguments are constrained to a valid JSON object;
`{}` is allowed and the declared parameter schema is not silently enforced.
Explicit strict functions retain their schema constraints. Automatic tool
choice constrains arguments when a tool call starts. The template communicates
required/named choices and explicit strictness, while grammar and sampler
changes enforce the protocol.

Required calls cannot finish successfully with reasoning alone. Malformed
normal-stop arguments fail the response; token exhaustion produces an
incomplete response, without a successful arguments-done event for malformed
or partial arguments. No malformed-JSON repair/retry loop is enabled.

The 2026-09-19 validation recorded 133 integrated CPU tests and eight GPU
ragged-prefix attention reference cases. Six final live protocol checks covered
strict unset/false/true, auto, reasoning accounting, and forced truncation;
two additional live checks covered conflicting assistant/tool constraints.
These checks do not establish all model capabilities or long-running stability.

Run the benign local c8 tool workload when there is capacity for a benchmark:

```bash
.venv/bin/python benchmarks/benchmark_qwen_tool_protocol.py \
  --url http://127.0.0.1:8000 --concurrency 8 --turns 2 --records 240 \
  --output perf/results/2026-09-19/qwen-tool-throughput/local-c8.json
```

Repeat after warm-up with a separate output filename. It sends about 10K input
tokens per first round and validates 16 rounds: successful terminal status,
required calls, valid arguments, batch identity, and delta/done equality.
Reported throughput is exact API output usage divided by workload wall time;
it includes thinking tokens. `first_event_s` is the first SSE event, not token
TTFT. Running this concurrently with production traffic changes the comparison.

| Warm run | Output tokens | Wall time | Output tok/s | Passed rounds |
| --- | ---: | ---: | ---: | ---: |
| Repaired tensor descriptors on | 21,510 | 219.83 s | 97.85 | 16/16 |
| Tensor descriptors off | 18,748 | 91.21 s | 205.54 | 16/16 |
| Tensor descriptors off repeat | 20,341 | 97.03 s | 209.63 | 16/16 |

This is a 2.10–2.14× improvement for that local inventory workload. There was
one warm baseline, clocks were not locked, and token sequences/lengths varied
between kernel paths. It is not a matched-step kernel speedup or a production
traffic guarantee. Exact-window audits had zero malformed completed arguments,
missing required calls, or delta/done mismatches. The TD path also received a
separate padding correctness fix: invalid logical-prefix K/V values are zeroed
so poisoned page tails cannot leak NaNs through `0*NaN`.

See `perf/optimization_status.md` for the full measurement record. Raw results
stay under ignored `perf/results/2026-09-19/{qwen-tool-throughput,
qwen-required-correctness,qwen38-attention-isolation}/` on the measurement host;
they are not included in commits.

## Other checkpoints and qualification limits

| Recipe | Status |
| --- | --- |
| `qwen38-hauhau-q8-b70-4` | HauhauCS Aggressive Q8_K_P GGUF, embedded MTP k=1, TP4 decode graphs. Earlier depth/throughput measurements support this configuration; it was not requalified in the FP8 attention comparison. The FastMTP-32K sidecar is not selected. |
| `qwen38-huihui-bf16-b70-4` | BF16 target/draft, TP4, FP8 KV. Explicitly gated `in-progress` after Xe copy-engine CAT faults/resets and a stalled benchmark; do not describe it as qualified. |
| Historical Qwen3.8 NVFP4 | The 2026-09-01 notebook records RadixArk NVFP4 TP4 measurements. No corresponding Qwen NVFP4 B70 registry profile is provided here; those short-prompt results are not comparable to this FP8 inventory run. |

The checkpoints differ in training/uncensoring as well as quantization. These
runs do not isolate FP8 versus Q8 versus BF16 model quality. Vision is present
in the FP8 source, but the 2026-09-19 benchmark is text/tool-only.

When investigating low throughput, collect aggregate request running/waiting
counts, context lengths, KV and prefix-cache utilization, MTP acceptance,
reasoning versus visible output usage, and per-step target/draft timing.
Interval logger TPS alone does not establish a workload regression. Check the
kernel journal for Xe engine resets, CAT faults, PCIe/AER events, and device
loss; `xpu-smi` does not expose all those fault counters. A healthy `/health`
response alone does not establish forward progress during a sampling RPC stall.
