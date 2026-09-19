# Qwen3.8 on four Intel Arc Pro B70 cards

The forward-ported FP8 recipe is `qwen38-uncensored-fp8-4`, using OrcaRouter's
Qwen3.8-27B checkpoint on four 32 GiB Intel Arc Pro B70 cards. **All three B70
recipes remain gated `in-progress` pending qualification on current main.**
The measurements below belong to the saved integration checkout, not this
forward port. No current-main native build or GPU serving test was available
while preparing these PRs.

Merge the dependencies in [the review map](qwen38-b70-review-stack.md) before
qualification. Copying `profiles.json` alone is insufficient. In particular,
PR #21's XPU foundation still needs integration with current main; merging its
historical tree wholesale would overwrite newer work. The saved B70 base also
contains the separate UR/XPTI tracing workaround `c6d985940`, absent from #21.
It clears tracing variables in each worker before device initialization. That
workaround is outside the 17-commit source stack and must be reviewed separately;
do not assume the new recipes reproduce it through shell environment settings.

## Registered recipes

| Profile | Weights | Native MTP | Status |
| --- | --- | --- | --- |
| `qwen38-uncensored-fp8-4` | Offline 128×128 block FP8 E4M3 | k=3 | Gated; historical sampling RPC stalls remain unresolved |
| `qwen38-hauhau-q8-4` | Q8_K_P GGUF | Embedded NextN, k=1 | Gated; fresh GGUF layout/model parity required |
| `qwen38-bf16-4` | BF16 safetensors | k=1 | Gated after historical Xe copy-engine CAT faults/resets |

These checkpoints differ in training as well as quantization. Their measurements
do not isolate precision effects. Preserve vision for the safetensors checkpoints;
text-only checks do not qualify their image paths. The separate FastMTP-32K
sidecar is not selected by the Q8 recipe.

The FP8 source is pinned to revision
`0f3cdb83820a8190ffedaef5b29cf4a635e49b4d` in
`orcarouter/Qwen3.8-27B-Uncensored-FP8`. Its served model ID is
`Qwen3.8-27B-Uncensored-FP8`. All three recipes use TP4, 262144 maximum context,
32 maximum sequences, 16384 batched tokens, target `fp8_e4m3` KV and draft
`fp8` KV. Context capacity does not promise 32 simultaneous full-context requests.

The FP8 recipe selects `linear_backend=xpu`, `FLASH_ATTN`, prefix caching,
`FULL_DECODE_ONLY` breakable graphs and token-width-aware capture sizes through
128. It disables Triton tensor descriptors and selects 16 softmax segments.
The checked-in Qwen tool template uses native thinking and tool tokens.
Do not add online `--quantization fp8` to this already serialized FP8 checkpoint.
Its uncalibrated KV scales default to 1.0, which still requires accuracy checks.

The profile supplies a 2,000-token thinking default. Explicit `-1` disables that
default; `null` inherits it under current-main semantics. Reasoning effort `none`
disables thinking. The effective budget is bounded by the response token ceiling;
leave room for the final answer or tool call. Main already prevents an MTP batch
with an in-budget `</think>` from receiving a second forced closing marker.

## Build and qualification

Use Linux Xe/Level Zero, Intel oneAPI DPC++ (`icpx`) and a matching PyTorch XPU
environment. The historical machine used torch `2.15.0.dev20260815+xpu`,
torchvision `0.30.0.dev20260816+xpu`, triton-xpu `3.8.0+git1e2d42a0`,
vllm-xpu-kernels `0.1.13` rebuilt locally, XGrammar `0.2.3` and
intel-sycl-rt `2026.1.0`. These are an environment receipt, not a binary lockfile.
The rebuilt dependency binary is not included in Git. Match the PyTorch ABI;
a previously mismatched binary registered operators on the wrong dispatch key.

Consult `requirements/xpu.txt` and `cmake/xpu.cmake` after integrating the XPU
foundation. Keep compiler setup confined to the build subshell:

```bash
(
  source /opt/intel/oneapi/setvars.sh
  VLLM_TARGET_DEVICE=xpu .venv/bin/python setup.py build_ext --inplace
)
.venv/bin/python -c 'import vllm._quixicore_C'
```

The XPU targets are `_quixicore_C` and `quixicore_xpu_ops`, rather than the CUDA
stable-ABI target. Keep both resulting artifacts together. Do not source oneAPI
compiler setup in the serving shell: it can load another SYCL runtime beside
PyTorch's runtime. Check that CUDA Triton has not shadowed triton-xpu.

After qualification and registry promotion, inspect the resolved recipe through
SlimServe. Both `--dry-run` and serving enforce the gate until then:

```bash
export SLIMSERVE_CACHE="$HOME/models"
export ZE_AFFINITY_MASK=0,1,2,3
.venv/bin/python -m slimserve.cli qwen38-uncensored-fp8-4 --dry-run
```

The qualification gate deliberately prevents normal serving until the registered
status is promoted after recorded native parity, clean cold boot, text/image and
tool canaries, cache reuse, matched throughput, and sustained mixed-arrival tests.
After that promotion, the serving command is:

```bash
.venv/bin/python -m slimserve.cli qwen38-uncensored-fp8-4 \
  --serve --host 127.0.0.1 --port 8001 --yes
```

SlimServe owns artifact downloads and verification. The FP8 source requires
Hugging Face access; authenticate on the deployment host. Inherited process
environment overrides profile environment, so remove stale experimental settings.
The [user service example](qwen38-b70.service.example) is for a qualified deployment
and uses a loopback backend for the optional
[streaming proxy](../../tools/serving_diagnostics/README.md).

## Historical bounded evidence

The source checkout recorded eight GPU ragged-tail reference cases and live
checks for strict unset/false/true tool policy, auto selection, reasoning usage,
forced truncation and conflicting assistant/tool constraints. Those checks are
historical; the forward port has CPU regression evidence listed per PR.

The local inventory benchmark sends required tool calls over two rounds, checking
terminal status, JSON syntax, batch identity and delta/done equality. It records
raw events and exact API usage; `first_event_s` is not token TTFT. Run it through
the registered serving path after qualification, with a new artifact directory:

```bash
.venv/bin/python benchmarks/benchmark_qwen_tool_protocol.py \
  --url http://127.0.0.1:8001 --model Qwen3.8-27B-Uncensored-FP8 \
  --concurrency 8 --turns 2 --records 240 \
  --output perf/results/b70-qualification/local-c8.json
```

| Historical warm run | Output tokens | Wall seconds | Aggregate output tok/s | Passed rounds |
| --- | ---: | ---: | ---: | ---: |
| Repaired tensor descriptors enabled | 21,510 | 219.83 | 97.85 | 16/16 |
| Tensor descriptors disabled | 18,748 | 91.21 | 205.54 | 16/16 |
| Descriptors disabled, repeat | 20,341 | 97.03 | 209.63 | 16/16 |

The 2.10–2.14× difference applies to that inventory workload. It used one warm
baseline, unlocked clocks and different output lengths/token sequences. It is
not a matched-step kernel speedup or a sustained production result. Original raw
files remained on the measurement host under ignored
`perf/results/2026-09-19/{qwen-tool-throughput,qwen-required-correctness,qwen38-attention-isolation}/`.
They are not included here.

Subsequent traffic stalled at a `sample_tokens` RPC timeout after about five
minutes. A restart stalled one rank inside Level Zero on a scalar allocation;
a device reset restored health. This was recovery, not a root-cause fix. No reset
is automated by the service example. The BF16 experiment separately encountered
Xe copy-engine faults. These unresolved failures justify keeping the profiles
gated. `/health` and interval TPS cannot establish model progress or soak success.

See [the protocol findings](qwen38-b70-protocol-followup.md) for the independently
confirmed streamed/final identity bug and the limits of its recorded follow-up.
