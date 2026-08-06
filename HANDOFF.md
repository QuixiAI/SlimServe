# Kimi K3 GGUF on MI300X — handoff

<!-- markdownlint-disable MD013 -->

Updated: 2026-08-06 UTC

## Resume here

Kimi K3 serves correctly on this box. **TP8 now beats TP6 at c8.** The
regression was the IQ2_XXS `w13` vector kernel expanding every route over the
full 6144 output rows even though expert parallelism makes 7/8 of the routes
non-local. The working-tree change adds an EP-aware, token-major kernel that
loops over the 16 routes inside each workgroup.

### Verified state

Measured at 1k in / 2k out, `--ignore-eos`, each run gated on the model
answering a known question first. Both coherence-checked 3/3 on arithmetic,
geography and general knowledge.

| profile | topology | c1 tok/s | c8 tok/s | TPOT c1 / c8 |
| --- | --- | --- | --- | --- |
| `k3-6` | TP6, tensor-parallel MoE | **34.0** | 120.4 | 28.74 / 62.85 ms |
| `k3-8` before fix | TP8, expert-parallel MoE | 31.1 | 94.8 | 31.45 / 80.98 ms |
| `k3-8` now | TP8, EP token-major `w13` | not rerun | **124.6** | not rerun / **60.78 ms** |

`k3-8` is now the fastest measured c8 profile: 3.5% more throughput and 3.3%
lower TPOT than TP6. Its c1 result has not been rerun on this build, so retain
the TP6 c1 number for latency comparisons. For reference, at the start of this
work TP6 managed 6.30 tok/s single stream and crashed outright at concurrency 8.

### How to run anything

```bash
source /home/hotaisle/SlimServe/.venv/bin/activate   # never system python/pip
```

Serving and one-shot chat:

```bash
slimserve k3-8                 # interactive
slimserve k3-8 --serve         # OpenAI endpoint
slimserve k3-8 -p "2 + 2?"     # one shot
slimserve --list               # legal profiles
```

Benchmarks and the coherence check live in the session scratchpad
(`/tmp/claude-1000/.../scratchpad/`), not in the repo:

- `bench_k3.py <profile>` — throughput sweep. `CONC=1,8` picks concurrencies;
  `BACKEND=HIP_MLA` forces an attention backend. **It refuses to report numbers
  until the model answers a known question**, which is the single most important
  guard here — see below.
- `coherence.py` — three known-answer questions against a profile.
  `PROFILE=` and `BACKEND=` env overrides.

### The rule that matters most

**`--ignore-eos` throughput cannot detect a broken model.** It generates a fixed
token count whatever the content, so a model emitting `!!!!!!!!` benchmarks at
full speed. This cost two iterations of reported "gains" on a broken build
before it was caught. The harness now gates on a correct answer; keep that gate,
and never report a number from a run that skipped it.

## Why TP8 used to lose to TP6 — attributed

Matched 120-second `rocprofv3` traces of rank 0 account for 17.90 ms of the
18.13 ms c8 TPOT gap (62.85 → 80.98 ms). The residual is 0.23 ms, well within
run noise. The profiled runs themselves were slower, so use the original
coherence-gated numbers above for throughput and the matched traces only for
kernel attribution.

### What is measured

| component | cost per decode step | how |
| --- | --- | --- |
| TP8 IQ2_XXS `w13` MoE vec | 37.87 ms | rank-0 kernel trace, 92 calls/step |
| TP6 IQ2_XXS `w13` MoE vec | 16.66 ms | matched trace, 92 calls/step |
| TP8 Q2_K `w2` MoE vec | 6.48 ms | rank-0 kernel trace, 92 calls/step |
| TP6 Q2_K `w2` MoE vec | 9.79 ms | matched trace, 92 calls/step |
| EP8 collectives | 10.08 ms | 8-GPU torchrun, 93 layers |
| TP6 collectives | 10.63 ms | same |
| **net traced MoE penalty** | **17.90 ms** | vs **18.13 ms** end to end |

### The governing fact

`moe_vec_q` launches one output-row workgroup for every `(token, route)` pair.
For eight tokens and top-k 16, TP8 launches 6144 × 128 = 786,432 workgroups per
MoE layer. Only about 1/8 of routes are local, so 688,128 workgroups read a `-1`
expert id, write zero, and return. TP6 tensor-shards `w13` to 1024 rows and all
routes are valid, so it launches only 1024 × 128 = 131,072 useful workgroups.

This geometry makes TP8 `w13` 230.59 µs/call slower: 21.21 ms over 92 MoE
layers. TP8's `w2` is 35.99 µs/call faster, recovering 3.31 ms. The net 17.90
ms is the regression. This is not an extra-launch or collective problem: the
MoE call counts match; it is excessive work inside the same `w13` calls.

The fix plan is `docs/tp8-performance-plan.md`. Keep `w2` on its current path;
it already wins on TP8. Change only the EP `w13` geometry: launch over tokens,
loop over top-k routes inside each workgroup, and skip non-local experts. The
new kernel writes zeros for skipped routes without launching a separate
workgroup for every route and output row.

That fix is implemented in the current working tree. At the exact live shape,
the route-major op took 406.89 µs and the token-major op 112.23 µs (3.63×), with
bit-identical output. The unchanged end-to-end harness passed its Paris gate and
measured 124.62 tok/s / 60.78 ms TPOT at c8, up from 94.8 / 80.98. The separate
three-question evaluation passed 3/3 (`4`, `Paris`, `The Pacific Ocean`).

### Dead ends — do not retry

Each was tested end to end with a number:

| idea | result |
| --- | --- |
| Hand-written HIP peer-to-peer all-to-all | EP8 collectives 10.08 ms vs TP6 10.63 ms — not the differentiator |
| `use_sequence_parallel_moe` at dp1 (drop the `dp > 1` gate) | TPOT 31.45 → 82.15 ms at c1. The gate is doing real work |
| `VLLM_GGUF_MOE_VEC_W2=0` (force MMQ tile for w2) | 31.14 → 28.88 tok/s at c1. The ROCm 128-row default is correct |
| Repack `w2` transposed for a finer MoE split | 384 vs 512 units/rank is 3–7%. Not worth a requantization |
| Pad intermediate 3072 → 4096 for tensor-parallel MoE at TP8 | Per-rank work becomes 512, identical to TP6, and work is not what costs |
| Vectorized (`dwordx4`) loads in the MLA kernel | Both layouts move 18 cache lines/row; instruction count is not the limit |

### Working profiler recipe

The original registration failure came from mixing PyTorch's bundled,
unversioned rocprofiler SDK/register libraries with the system ROCm 7.2.4
profiler. The working setup is:

1. Build two empty shared-library shims whose SONAMEs are the unversioned names
   PyTorch requests and whose `NEEDED` entries point to the system
   `librocprofiler-sdk.so.1` and `librocprofiler-register.so.0`.
2. Preload the register shim before the SDK shim, set `ROCP_TOOL_ATTACH=1`, and
   start the benchmark normally. This loads
   `/opt/rocm/lib/librocprofiler-sdk-attach.so` before HSA initialization.
3. With Yama `ptrace_scope=1`, have each worker call
   `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` at startup. A temporary
   `sitecustomize.py` on `PYTHONPATH` is sufficient.
4. After CUDA/HIP initialization, identify rank 0 and attach:

   ```bash
   rocprofv3 --attach <rank-0-pid> --attach-duration-msec 120000 \
     --kernel-trace --output-format csv --output-directory <trace-dir>
   ```

Attaching before `torch.cuda.init()` returned no trace; direct launch reproduced
the registration error. The successful TP8 trace has 7,096,093 dispatches over
120.111 seconds; the matched TP6 trace has 7,963,792 over 119.285 seconds.
Synthetic MoE microbenchmarks remain unsuitable because random routing changes
which experts' weights are touched.

## What changed in this work

Uncommitted current working tree:

- Attribute the entire TP8 regression with matched rank-0 kernel traces.
- Add the EP token-major IQ2_XXS `w13` kernel and route only EP `w13` calls to it.
- Add a mixed local/non-local expert correctness test to the existing GGUF
  vector test file.
- Validate 3/3 coherence and c8 at 124.62 tok/s / 60.78 ms TPOT.

Validation run:

- Focused EP kernel test: 1 passed.
- `tests/model_executor/test_kimi_k3_ep.py`: 11 passed.
- Selected pre-commit hooks: all passed, including Ruff, mypy, clang-format,
  markdownlint, SPDX, and configuration checks.
- The full `test_gguf_vec_writes_all_outputs.py` run was 8 passed / 7 failed in
  its older dequant-reference cases. Those cases feed arbitrary bytes to quant
  formats and produce pathological scales; the new EP test passes. Do not claim
  the whole file is green.

Committed, newest first:

- `9d70023` phase 0 needs a profiler, not microbenchmarks
- `a70bf06` a measured plan for making TP8 beat TP6 → `docs/tp8-performance-plan.md`
- `e9fba4a` K3 decode is latency-bound, so more GPUs cannot shorten a step
- `da84a78` the TP8 padding fix would buy nothing
- `edef4b9` **k3-8 means TP8** (was DP4×TP2; that profile is dropped)
- `ffb2e82` **TP8 works, with the experts kept whole** (expert parallelism)
- `298acb4` prove the TP8 MoE break; fix a test that could not fail
- `3fce38c` withdraw the TP8 profile (later restored correctly)
- `bd3967b` **drop the GGUF output pre-fill from the decode matmul** (~20%/call)
- `911d145` **MLA: 4 KV tokens per iteration, and a measured split count**
- `839f10f` **a HIP MLA decode kernel** — head count as a grid dimension
- `fd760db` let a backend decline a head count instead of asserting
- `98adf79` cap `max_num_seqs` to the KDA state slots
- `9474c1c` **delete `enforce_eager`** from the fork entirely
- `d9fd45a` stop serving K3 eager

### The HIP MLA decode kernel

`csrc/quixicore/rocm/mla_decode_kernels.cuh` + `csrc/quixicore/tm_rocm/qc_rocm_mla.cu`,
exposed as `HIP_MLA` in `vllm/v1/attention/backends/mla/hip_mla.py`, first in
ROCm's MLA priority list.

Why it exists: AITER's gfx942 MLA decode ships as pre-assembled code objects
with the query head count baked in, so only multiples and divisors of 16 run.
TP8 gives 12 heads per rank. This kernel takes the head count as a grid
dimension, so any TP size that divides K3's 96 heads works.

Performance: **parity with AITER's hand-written assembly** at 16 heads per rank
(32.46 vs 32.81/33.27 tok/s at c1, 90.67 vs 91.70/91.81 at c4) — it ties where
AITER works and runs the shapes AITER cannot. Validated against a float64
reference at 12/16/48 heads, shuffled block tables, split-K, and the 960-token
page K3's hybrid cache uses (`tests/kernels/test_mla_decode_gfx942.py`).

Design notes worth keeping: no MFMA (at 12 heads it is ~23 FLOP/byte against a
~246 balance point, and the 16-wide tile is what makes head counts rigid); no
branch on the nope/rope split (k_pe is rotated at insert, so the score runs over
all 576 lanes and the accumulate stops at 512); `max_seq_len` comes from the
metadata builder, never `seq_lens.max()`, which would sync the decode path and
break graph capture.

## Environment traps

These each cost real time. All are load-bearing.

**Header edits do not trigger a rebuild.** The ROCm build tracks no header
dependencies. Editing only a `.cuh` leaves ninja with "no work to do" and the
build still exits 0, shipping the previous binary. `touch`ing the `.cu` is *not*
enough — it recompiles against the stale hipified header in the build tree.
Delete both, rebuild, and grep the regenerated header for a token unique to your
edit:

```bash
cd build/temp.linux-x86_64-cpython-312
rm -f CMakeFiles/_quixicore_C.dir/csrc/quixicore/tm_rocm/qc_rocm_mla.hip.o \
      csrc/quixicore/rocm/mla_decode_kernels.cuh
ninja _quixicore_C
grep -c <your-token> csrc/quixicore/rocm/mla_decode_kernels.cuh
```

**Install a rebuilt `.so` with `mv`, not `cp`**, when a server may have it
mapped — `cp` writes in place and can crash the running process.

**Eager microbenchmarks lie about launch-bound kernels.** The serving path is
`cudagraph_mode: FULL_DECODE_ONLY`, so measure inside a `torch.cuda.CUDAGraph`.
K3's KDA kernels measured 78 µs eager and 4.1 µs in-graph — a 19× error that
pointed at the wrong target. If eager and in-graph differ wildly, the kernel is
launch-bound and its eager number is meaningless.

**Never `pkill -f <pattern>` or kill the output of a bare `pgrep -f`.** Each
Bash tool call runs in a wrapper shell whose command line contains the command
text, so the pattern matches the caller and kills it (exit 144). The same breaks
wait-loops: `while pgrep -f "bench_x.py"` never exits because the heredoc that
created the script is still on the spawning wrapper's command line. Collect PIDs,
verify none is a `shell-snapshots/snapshot-bash-*.sh` wrapper, then kill by PID.

**Killing a benchmark driver does not free the GPUs.** `slimserve.server.Server`
spawns `vllm serve` as a child that survives, holding all eight cards at 97%, so
the next run dies with "Free memory on device cuda:N (0.79/191.98 GiB)". After
stopping a run, check `rocm-smi --showmemuse` and kill surviving `api_server` /
`VLLM::*` PIDs explicitly.

**Auditing kernel output coverage.** To prove a kernel writes its whole output
(the prerequisite for deleting a defensive `fill_`), recompile that entry
point's fill as `quiet_NaN()` and look for survivors. Two weaker versions both
returned false "clean" results and shipped a broken model: testing entry point A
while B's zero-fill was still compiled in (the fill zeroed the buffer, so the
result measured the fill); and poisoning a tensor, freeing it, and hoping the
caching allocator returned the same block. Also use **zeroed weights** — random
bytes decode to NaN K-quant scales and produce NaN *outputs* indistinguishable
from unwritten memory.

## Why TP8 needs expert parallelism

`k3-8` sets `enable_expert_parallel`. This is not optional and not a leftover.

Tensor-sharding the MoE splits each expert's `w2` along its **packed byte**
axis. Traced from a live TP8 load (`VLLM_TRACE_MOE_SHARD` in
`_gguf_moe_weight_loader`), expert 0:

```text
w1  src=(896, 3072, 924)   dst=(896, 768, 924)
w2  src=(896, 3584, 1008)  dst=(896, 3584, 126)
w3  src=(896, 3072, 924)   dst=(896, 768, 924)
```

`_materialize_gguf_moe_param` divides that byte axis by `tp_size`: 1008 / 8 =
126, against a Q2_K `type_size` of 84 — **1.5 blocks**, so every rank starts
decoding mid-block and the model emits `!!!!!!!!`. TP2/4/6 give 6/3/2 whole
blocks, which is why only TP8 broke. `w1`/`w3` split dim 1 in *elements* and
leave their byte axis whole, so they are safe at any size.

Beware: `create_weights` sets `input_dim: 1` on w2, which describes the logical
layout, not the packed one. Reading that attribute alone suggests the split is
safe. It is not — trace it.

Expert parallelism sidesteps this entirely: each rank holds 112 of the 896
experts whole, so no quant block is ever cut.

## TP6 baseline status

**Working end to end.** SlimServe loads the 858 GB
`Kimi-K3-IQ2_XXS-Q2_K.gguf` with the BF16 vision projector on six MI300X
GPUs, profiles the model, allocates the hybrid cache, renders K3's native XTML
chat format, and generates coherent text and vision responses.

The final TP6 validation completed with exit code 0. Greedy outputs included:

```text
2 + 2       -> 4
capital of France -> Paris.
```

The raw offline output includes K3's `<think>` and `<response>` XTML framing;
the semantic corruption seen earlier is gone.

A sequential three-user-turn multimodal validation also completed with all five
requested images, preserved assistant history, and exit code 0. The model
correctly identified both landscapes, the weathered bench scene, the family of
five, and three people running.

The numbers in this section predate the MMQ expert-id fix, so they describe a
TP6 whose prefill was dropping every expert above 255. See the re-run recorded
under "A second, independent defect" above for current behaviour.

The source changes and this handoff are published together for human review.

## Tested invocation

The fork's global cache default is FP8 for GLM, so K3 must explicitly request
the native cache dtype with `kv_cache_dtype="auto"`. `HSA_XNACK=1` is also
required for this load.

```bash
cd /home/hotaisle/SlimServe
HSA_XNACK=1 \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_GGUF_MMPROJ=/home/hotaisle/models/antirez-kimi-k3-gguf/mmproj-BF16.gguf \
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5 \
.venv/bin/python -u -c '
from vllm import LLM, SamplingParams

model = "/home/hotaisle/models/antirez-kimi-k3-gguf/Kimi-K3-IQ2_XXS-Q2_K.gguf"
llm = LLM(
    model=model,
    tensor_parallel_size=6,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    gpu_memory_utilization=0.95,
    block_size=64,
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    mm_encoder_tp_mode="data",
    kv_cache_dtype="auto",
)
conversations = [
    [{"role": "user", "content": "What is 2 + 2? Reply with only the answer."}],
    [{"role": "user", "content": "What is the capital of France? Reply briefly."}],
]
outputs = llm.chat(
    conversations,
    sampling_params=SamplingParams(max_tokens=64, temperature=0),
)
for output in outputs:
    print(output.outputs[0].text)
'
```

Observed final-run facts:

- weight load: about 156–160 seconds per rank;
- model memory: 142.23 GiB per rank;
- native/BF16 hybrid cache: 12.48 GiB;
- cache capacity: 258,560 tokens;
- reported concurrency: 63.12 requests at 4,096 tokens;
- two prompts: about 18 input tokens/s and 10–11 output tokens/s combined;
- process exit: 0 (with the existing shared-memory shutdown warning).

## Three-turn, five-image validation

The five source images were downloaded and decoded with PIL before the model
run:

1. `https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png`
2. `https://picsum.photos/seed/picsum/200/300`
3. `https://picsum.photos/id/32/512/512`
4. `https://www.wolframcloud.com/obj/resourcesystem/images/a0e/a0ee3983-46c6-4c92-b85d-059044639928/6af8cfb971db031b.png`
5. `https://s3.amazonaws.com/cms.ipressroom.com/338/files/201808/5b894ee1a138352221103195_A680%7Ejogging-edit/A680%7Ejogging-edit_hero.jpg`

Their visual-token counts were 1,820, 88, 361, 1,386, and 375 (4,030 total),
so this test used `max_model_len=8192` with
`max_num_batched_tokens=4096`. It made three sequential greedy `LLM.chat`
calls with `chat_template_content_format="openai"` and `thinking=False`,
appending each generated assistant reply before the next user message:

- turn 1: images 1–2, compare the landscapes;
- turn 2: images 3–4, describe the subjects and identify the family;
- turn 3: image 5, count and describe the people and recall the family image.

Observed results:

- Turn 1: 2,000 prompt tokens, 74 output tokens, `stop`; Yosemite river valley
  versus snowy mountain.
- Turn 2: 3,898 prompt tokens, 45 output tokens, `stop`; weathered wall/bench
  and a family of five in the second new image.
- Turn 3: 4,391 prompt tokens, 39 output tokens, `stop`; three people running
  and correct recall of the preceding turn's family image.

The corrected run took about 14.2, 9.8, and 9.7 seconds for the three requests
after model initialization. It used the MIOpen vision patch-embedding fallback
and completed without a vision-kernel failure.

## TP6 and AITER's custom all-reduce

At world size 6, AITER's custom all-reduce reproducibly illegal-addresses on a
`[5, 7168]` BF16 collective. The communicator change disables only that AITER
collective for world size 6; the working run uses the normal vLLM/PyNCCL path
while retaining AITER MLA and MoE kernels.

## Root semantic defects

### 1. The chunk KDA prefill kernel is wrong on gfx942

The AMD chunked KDA path produced numerically unrelated results on MI300X:

```text
chunk vs direct recurrence: corr 0.1567
chunk max output:           0.04736
reference max output:       0.0008507
```

The generic fused recurrent KDA kernel agrees with a direct mathematical
recurrence:

```text
output max error: 9.31e-10
state max error:  1.19e-7
correlation:      1.0
```

On ROCm gfx942, prefill now materializes K3's gate and beta and uses the
verified recurrent kernel. Other platforms retain the chunk path. The generic
kernel's state-index stride was also fixed so one expanded cache index can be
reused across every token in a packed prompt. Packed decode separately matches
the direct recurrence exactly for output and within `2.98e-8` for state.

### 2. Latent MoE normalized TP partials before reducing them

Each TP rank computes a partial 3,584-wide routed-expert result. The old runner
applied K3's RMSNorm and latent up-projection to each partial, then all-reduced
the full-width outputs. The released reference first sums the latent partials
and only then normalizes. RMSNorm is nonlinear, so these are not equivalent.

`KimiRoutedOutputTransform` now marks that it requires a reduced input, and the
MoE runner reduces routed and shared partials before applying that transform.
A focused six-part oracle has `new_max_error = 0.0`; the former order differs
from the reference by `4.118` on the same test values.

### 3. vLLM rejected K3's native chat renderer

K3 renders XTML inside `TikTokenTokenizer.apply_chat_template`; it does not
ship a Jinja template. vLLM rejected chat before calling the override. The
GGUF tokenizer now exposes a non-empty sentinel template so resolution reaches
the native renderer. Its 102-token test prompt matches the released tokenizer
exactly.

### 4. The mmproj Q/K rows were in llama.cpp's split 2D-RoPE layout

The first real-image run was mechanically stable but described every photo as
a repeated floral strip. Image preprocessing was bit-identical to the released
K3 processor, and the patch embedding and projector tensors were bit-identical
to safetensors. The fused `v.blk.N.attn_qkv.weight` tensors isolated the fault:
llama.cpp had permuted Q and K from K3's native interleaved 2D-RoPE order into
its split `[x | y]` order, while the loader treated them as pure renames.

The adapter now restores native interleaved Q/K rows while leaving V unchanged.
All 27 corrected fused QKV tensors are bit-identical to the released K3
safetensors; block 0's pre-fix maximum error was `2.34375`, and its post-fix
error is `0.0`. The full three-turn image run above then produced the expected
semantics.

### 5. The GGUF MoE kernel was handed AITER's residency mask under EP

`RoutedExperts.expert_map` returns AITER's 0/1 `expert_mask` instead of the
global-to-local map whenever AITER's fused MoE is enabled. The GGUF kernel
indexes its local expert stack with that tensor, so under EP every routed token
went to local expert 0 or 1. `global_to_local_expert_map` now returns the real
map unconditionally. See "The bug that caused the incoherence" above for the
oracle numbers.

### 6. The MMQ tile kernel refused expert ids above 255

Inherited from upstream's GGUF MoE kernel. Kimi has 896 experts, so 71% of them
were silently zeroed in every prefill-width MoE call that saw global ids -- the
TP6 configuration. See "A second, independent defect" above.

## Other required working-tree fixes

- `vllm/models/kimi_k3/common/mm_preprocess.py`: build preprocessing from the
  mmproj instead of decoding the 858 GB text GGUF as JSON.
- `vllm/model_executor/model_loader/gguf_adapters/kimi_k3.py`: map all text and
  vision weights, restore native vision Q/K row order, and derive unquantized
  fused attention parents.
- `vllm/model_executor/models/registry.py`: register
  `KimiLinearForCausalLM`.
- `vllm/model_executor/layers/mla.py`: carry and apply K3's MLA output gate.
- `vllm/model_executor/layers/vocab_parallel_embedding.py`: pad the
  163,840-token vocabulary to a size divisible by TP6.
- `vllm/models/kimi_k3/amd/linear.py`: use GGUF methods for quantized latent
  projections and declare the nonlinear pre-reduction requirement.
- `vllm/models/kimi_k3/amd/model.py`: keep the BF16 vision tower/projector
  outside GGUF text quantization.
- `vllm/model_executor/layers/quantization/gguf/fused_moe.py`: pass K3's SITU
  beta 4.0 and linear beta 25.0 into GGUF MoE activation.
- `vllm/v1/attention/backends/gdn_attn.py`: resolve the generic GDN prefill
  backend without importing a removed Qwen module.
- `vllm/distributed/device_communicators/cuda_communicator.py`: avoid the
  broken AITER custom all-reduce at world size 6.
- `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`: use the
  verified recurrent KDA prefill on gfx942.
- `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`: honor the
  token stride in continuous-batching state-index maps.
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`: reduce latent-MoE
  partials before nonlinear output transforms.
- `vllm/transformers_utils/gguf_kimi_k3.py`: allow vLLM chat resolution to
  invoke K3's native XTML renderer.

## Verified model data

- All operative GGUF config fields and both layer lists match the released
  `config.json`: 93 layers, 69 KDA + 24 MLA, 896 experts, top-16, 96 heads.
- GGUF and released tokenizers produce identical IDs on text, code, CJK,
  whitespace, emoji, special tokens, and the final native chat prompt.
- All 2,736 text tensors map, including 276 expert stacks; all 168 mmproj
  tensors map.
- The fused image processor matches the released processor exactly on all five
  validation images, including resized grids and every normalized pixel value.
- All 27 vision QKV tensors match safetensors exactly after split-to-interleaved
  row restoration.
- Representative BF16/F32 attention and config tensors are bit-identical to
  safetensors. Representative Q8 dequantized weights correlate above 0.999985.
- SITU kernels agree with their high-precision oracle.
- The attention-residual Triton kernel agrees with the PyTorch reference.
- KDA gate preprocessing agrees with direct PyTorch (`1.83e-4` maximum gate
  error on values with mean magnitude about 99.5; beta error `1.19e-7`).

## Files and checkpoint facts

```text
/home/hotaisle/models/antirez-kimi-k3-gguf/
  Kimi-K3-IQ2_XXS-Q2_K.gguf
  mmproj-BF16.gguf
  mmproj-F16.gguf
  mmproj-F32.gguf

/home/hotaisle/models/Kimi-K3/
  config.json
  tiktoken.model
  model-00001-of-000096.safetensors ... model-00096-of-000096.safetensors
```

The served GGUF has the published erroneous `kimi-k3.vision = false` header
byte patched to true. The file still omits K3 metadata for SITU beta, SITU
linear beta, and attention-residual block size; the config parser warns and
uses the released values. The BF16 mmproj is preferred because the source
vision weights are BF16.

## Remaining caveats

- The run emits a pre-existing missing `lora_hf_hub_resolver` plugin error but
  continues with the available plugins.
- Shutdown warns about one leaked shared-memory object; the validated process
  still exits 0.
- The vision patch embedding falls back to MIOpen because this AITER build has
  no Triton `conv2d`. The three-turn/five-image test completed on this fallback,
  but upstream describes the MIOpen path as intermittently failing under load.
- K3's GGUF-native chat output is XTML-framed in offline `LLM.chat`. Confirm the
  desired reasoning/content parser settings when exposing the OpenAI endpoint.
- Full mypy was not run. Final ruff/compile/diff checks are recorded at handoff
  time after the last source change.
- The MMQ expert-id fix lives in a `.cuh`, so a fresh clone must rebuild the
  stable-ABI extension before it takes effect; a stale `_C_stable_libtorch`
  silently reinstates the 255-expert ceiling.
- Both fixes affect any GGUF MoE checkpoint with more than 256 experts, not
  only Kimi K3. DeepSeek-V4 (256 experts) sits exactly at the old boundary and
  was unaffected; GLM-5.2 has 160.
- No duplicate-work/PR checks were run because no PR to upstream was proposed
  or opened. A human must review every changed line and run the relevant tests
  before submitting anything upstream.
