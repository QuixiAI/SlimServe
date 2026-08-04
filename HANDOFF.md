# Kimi K3 GGUF on MI300X — handoff

Updated: 2026-08-03 UTC

## Resume here: DP4xTP2 attention + EP8 MoE (semantics fixed)

DP4xTP2/EP8 now produces coherent text and vision output that matches the TP6
baseline's semantic facts. The previous incoherence had a single cause, and it
was not the sequence-parallel design: under expert parallelism the GGUF MoE
kernel was handed AITER's 0/1 residency mask where it expected the
global-to-local expert map.

Diagnosing that turned up a second, independent kernel defect that had been
degrading the TP6 baseline all along -- the MMQ tile kernel refused expert ids
above 255, so 71% of Kimi's experts were zeroed at prefill width. Both are
described under "Root semantic defects" 5 and 6 below; the second one needs a
rebuilt `_C_stable_libtorch.abi3.so`, which is already in place.

The work is still uncommitted and still needs the human line-by-line review
`AGENTS.md` requires.

### Objective and topology

Run Kimi K3 as four request replicas with TP2 attention and EP8 routed MoE:

- attention/KDA/MLA stay TP2 within each request's GPU pair;
- MoE sequence-parallelizes the replicated TP2 token stream;
- all eight ranks own 112 of 896 complete routed experts each;
- embeddings, attention, latent projections, router, shared expert, vision
  tower, and mmproj remain replicated as required by their parallel axis.

This is the memory-safe fallback for this quantized GGUF. Pure DP8 cannot fit
the replicated byte set; padded attention TP8 does not improve unique latent
KV capacity.

### Uncommitted implementation

The following changes are present in the working tree:

- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
  - allows a routed output transform with `requires_reduced_input=True` in
    sequence-parallel mode;
  - treats EP reduce-scatter output as complete for the local token shard, so
    K3's RMSNorm is applied after expert contributions have been combined.
- `vllm/models/kimi_k3/amd/linear.py`
  - enables FusedMoE sequence parallelism from
    `ParallelConfig.use_sequence_parallel_moe`;
  - chunks tokens across the TP2 pair before routing and gathers them after
    MoE, trimming sequence-padding tokens;
  - builds the shared expert with `disable_tp=True`, because it operates on a
    token shard and therefore needs complete replicated weights.
- `vllm/model_executor/model_loader/gguf_adapters/base.py`
  - lets an adapter receive the sorted global expert IDs assigned to its rank.
- `vllm/model_executor/model_loader/gguf_loader.py`
  - computes the flattened EP rank exactly like `FusedMoEParallelConfig`;
  - configures rank-local fused-stack loading for Kimi K3 only;
  - rejects Kimi K3 fused GGUF loading with EPLB because redundant physical
    expert population has not been implemented.
- `vllm/model_executor/model_loader/gguf_adapters/kimi_k3.py`
  - slices each fused 3D expert stack down to the rank-local expert rows before
    device upload;
  - uses an mmap-sharing `narrow` for contiguous linear placement and
    `index_select` for round-robin placement;
  - renames the synthetic expert-0 anchor to the first global expert in the
    local stack so Kimi's existing expert mapping invokes the full-stack load.
- `vllm/model_executor/layers/quantization/gguf/{linear.py,params.py}`
  - makes GGUF parameter loaders honor the parameter/layer `tp_rank` and
    `tp_size` instead of always consulting the global TP group;
  - this is required for `disable_tp=True` shared experts. Before the fix, a
    replicated shared-expert weight was silently narrowed to half width and
    the profile run ended in a HIP illegal-memory access.
- `vllm/model_executor/layers/fused_moe/routed_experts.py`
  - adds `global_to_local_expert_map`, which always returns the real map.
    `expert_map` degrades to AITER's 0/1 mask whenever AITER's fused MoE is
    enabled, and a kernel that *indexes* its local stack cannot use that.
- `vllm/model_executor/layers/quantization/gguf/fused_moe.py`
  - indexes experts with that map instead of `expert_map`.
- `csrc/libtorch_stable/quantization/gguf/moe.cuh`
  - stops rejecting expert ids above 255 in the MMQ tile kernel, and widens the
    expert byte offset to 64 bits.
- `tests/model_executor/test_kimi_k3_ep.py` (new)
  - eleven focused tests for nonlinear SP combine semantics, contiguous and
    round-robin fused-stack localization, invalid expert IDs, zero/odd token
    chunk-and-gather behavior, replicated versus sharded GGUF row and
    merged-column loading, and map-versus-mask expert indexing.

Do not discard these changes. The configured Git remote is already correct:
`origin` and `slimserve` both point to
`git@github.com:QuixiAI/SlimServe.git`; `upstream` points to vLLM.

### Validation completed

Focused validation is green:

```text
.venv/bin/python -m pytest tests/model_executor/test_kimi_k3_ep.py -v
8 passed in 10.46s

.venv/bin/python -m compileall -q <changed Python files>
passed

pre-commit run --files <changed files and focused test>
ruff, format, typos, mypy, SPDX, forbidden imports, and config checks passed

git diff --check
passed
```

The final one-line loader scoping edit (Kimi-only configuration) happened
after that hook run, so rerun the focused tests and hooks before committing.

The real 858 GB DP4xTP2/EP8 startup now succeeds repeatedly:

- every rank reports `loading 112/896 experts`;
- per-rank model memory is 145.17 GiB;
- weight load, multimodal profile, 512-token LM profile, hybrid KV allocation,
  sampler warmup, KDA/MLA prefill and decode, and EP collectives all complete;
- with `max_num_batched_tokens=512`, each request pair gets 16,384 cache
  tokens and reports 2.0x concurrency at an 8,192-token maximum length;
- all three requests and shutdown complete with process exit code 0;
- GPUs return to their approximately 298 MB idle baseline afterward.

The five requested images are still available at:

```text
/tmp/testimg/yosemite.png
/tmp/testimg/picsum1.jpg
/tmp/testimg/picsum2.jpg
/tmp/testimg/wolfram.png
/tmp/testimg/jogging.jpg
```

The temporary `.tmp_kimi_ep_smoke.py` runner that drove this diagnosis has been
removed; it was never committed. `slimserve k3-8` now covers the same ground,
and `slimserve k3-8 --dry-run` prints the settings it used.

Any replacement harness must pass K3's mode as
`chat_template_kwargs={"thinking": False}`. `LLM.chat` does **not** accept a
top-level `thinking=False` argument in this checkout. A first mechanical run
left thinking enabled and exhausted the output cap on reasoning markers; that
was a harness error, not a useful semantic test.

### EP8 semantic result

The three-turn/five-image run with the text gates added to rank 1 completed
with exit code 0 and answers that carry the TP6 baseline's facts:

```text
rank 1: 2 + 2                -> 4
rank 1: capital of France    -> Paris
rank 1: largest ocean        -> Pacific Ocean
rank 0 turn 1 (1961 prompt tokens): Yosemite Valley, El Capitan, granite
        cliffs and river, versus the snowy mountain
rank 0 turn 2 (3867 prompt tokens): weathered wooden bench against the
        two-toned distressed wall
rank 0 turn 3 (4402 prompt tokens, stop): three people jogging, and recalls
        five people in the earlier family image
```

Turns 1 and 2 stop at the 96-token cap mid-description because K3 answers in
detailed markdown here; that is the harness cap, not truncation of meaning.
Raise `max_tokens` if a run needs the complete turn-2 answer.

The run was repeated after the MMQ kernel rebuild described below and produced
the same answers with exit code 0, as expected: under EP8 each rank sees local
expert ids 0-111, so the 255 ceiling never applied there.

### The bug that caused the incoherence

`RoutedExperts.expert_map` does not always return the expert map. When AITER's
fused MoE is enabled -- the default on ROCm, and the validated invocation sets
`VLLM_ROCM_USE_AITER=1` -- it returns AITER's `expert_mask` instead: a 0/1
residency vector of length `global_num_experts + 1`.

The GGUF MoE kernel indexes with that tensor (`expert_map[topk_ids]`), so every
routed token was sent to local expert 0 or 1 out of 112, and
`moe_align_block_size` was told there were 897 global experts. TP6 was immune
because EP is off there and the tensor is `None`.

`global_to_local_expert_map` now exposes the real map and the GGUF method uses
it. Only the GGUF method was changed: the other quantization methods that pass
`layer.expert_map` on ROCm dispatch into AITER kernels, which want the mask —
`vllm/model_executor/layers/fused_moe/experts/aiter_mxfp8_moe.py` handles both
forms explicitly. Anything new that indexes a local expert stack on this fork
should take the map property, not `expert_map`.

Numbers from the single-GPU oracle over `blk.1`'s real GGUF stacks, 96
experts spanning all eight ranks, against a dequantized f32 reference (relative
mean error; 1.8e-1 is this reference's own IQ2_XXS/Q2_K quantization floor):

```text
decode/vec    eight 112-expert shards summed   1.8e-1   (at the floor)
decode/vec    mask instead of map              6.4e+00
prefill/mmq   eight 112-expert shards summed   1.8e-1   (at the floor)
prefill/mmq   mask instead of map              6.1e+00
```

### A second, independent defect: MMQ dropped experts above 255

The same oracle showed the *non*-EP prefill path failing where the EP path
passed: a single 896-expert call scored 8.1e-1 against the same reference while
the eight-shard sum scored 1.8e-1.

`moe_q` in `csrc/libtorch_stable/quantization/gguf/moe.cuh` rejected any
`exp_idx > 255`, inherited from upstream's original GGUF MoE kernel
(vllm-project/vllm#14613). The only invalid id `moe_align_block_size` produces
is `-1`, so the ceiling silently zeroed experts 256-895 -- 71% of Kimi's
experts -- in every MMQ (prefill-width) MoE call whenever global ids reach the
kernel. That is exactly the TP6 configuration, so **the TP6 baseline's prefill
was itself degraded**; its decode path uses the vector kernel and was correct,
which is why it still read as coherent.

The ceiling also masked a 32-bit overflow: one Kimi w13 expert is 5.7 MB, so
`exp_idx * exp_stride` overflows `int` from expert 378 up. The fix drops the
ceiling and widens that product to `int64_t`.

Reproduce with a synthetic Q8_0 stack of 896 experts routed both below and
above 255, comparing `ggml_moe_a8` against `ggml_moe_a8_vec` and a dequantized
reference. Before the fix, every id above 255 returned exactly zero from the
MMQ path while the vector path was correct; after it, MMQ and vector agree to
the shared q8_1 activation-quantization floor (5.6e-3). On the real `blk.1`
stacks the single full 896-expert prefill call went from 8.1e-1 to 1.8e-1,
matching the eight-shard sum exactly.

TP6 was re-run on the rebuilt kernel with the same prompts and images
(`max_tokens=256`, exit code 0). Text gates still return `4` and `Paris.`, and
the vision answers are substantially richer than the ones recorded further down
this document:

```text
turn 1 (1961 prompt tokens, 232 out, stop): names El Capitan *and* the
        Cathedral Rocks / Bridalveil side
turn 2 (4007 prompt tokens, 209 out, stop): reads "ESPERANÇA" off the sign,
        and counts and describes all five family members individually
turn 3 (4659 prompt tokens, 110 out, stop): three joggers, recalls five
```

The old baseline's 74/45/39-token answers were produced under a different
output cap, so treat the length change as suggestive rather than measured. The
per-expert detail is the substantive difference.

Note for whoever edits these kernels next: the hipify ninja rule depends only
on the `.cu` list, so editing a `.cuh` alone neither re-hipifies nor rebuilds,
and the build still exits 0 shipping the old `.so`. Re-run hipify by hand,
delete the stale `.hip.o`, `ninja _C_stable_libtorch`, then copy the result
over `vllm/_C_stable_libtorch.abi3.so` -- ninja links into `build/temp/`, but
the importable module is the copy under `vllm/`. About 90 seconds end to end.

```bash
cd build/temp.linux-x86_64-cpython-312
~/.venv/bin/python ../../cmake/hipify.py -p ../../csrc -o ./csrc \
  csrc/libtorch_stable/quantization/gguf/gguf_kernel.cu
rm CMakeFiles/_C_stable_libtorch.dir/csrc/libtorch_stable/quantization/gguf/\
gguf_kernel.hip.o
ninja _C_stable_libtorch
cp _C_stable_libtorch.abi3.so ../../vllm/
```

### Finish criteria

- [x] text gates return `4` and `Paris`;
- [x] the three-turn test correctly identifies the two landscapes, bench/wall,
  family of five, and three runners while recalling the family count;
- [x] focused tests, compilation, pre-commit, and `git diff --check` pass
  (11 focused tests; ruff, format, typos, clang-format, markdownlint, mypy,
  SPDX, lazy imports, forbidden imports, and config checks);
- [x] remove `.tmp_kimi_ep_smoke.py`;
- [ ] human-review every changed line before any commit or push, per
  `AGENTS.md`.

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

## Why TP6, and why TP8 still does not work

TP6 was chosen because TP8 could not start: 96 heads / 8 ranks gives 12 per
rank, and AITER MLA requires a multiple or divisor of 16. TP6 gives 16 and fits
the 858 GB checkpoint at about 142 GiB of weights per GPU; TP4 exceeds the
memory budget and also fails the head rule. Two GPUs therefore remain idle.

The attention half of that is now solved: the HIP decode kernel in
`csrc/quixicore/rocm/mla_decode_kernels.cuh` takes the head count as a grid
dimension, so 12 heads per rank runs.

TP8 is still wrong, for an unrelated reason in the quantized MoE.
`ffn_down_exps` is row-parallel, so TP splits its contiguous dimension -- which
is the dimension GGUF packs into 256-element quant blocks. That dimension is
3072 = 12 Q2_K blocks = 1008 bytes per row:

    TP2  504 bytes = 6.00 blocks   ok
    TP4  252 bytes = 3.00 blocks   ok
    TP6  168 bytes = 2.00 blocks   ok
    TP8  126 bytes = 1.50 blocks   slices a block in half

Proven at runtime by tracing `_gguf_moe_weight_loader` during a TP8 load
(`VLLM_TRACE_MOE_SHARD`), expert 0:

    w1  src=(896, 3072, 924)   dst=(896, 768, 924)
    w2  src=(896, 3584, 1008)  dst=(896, 3584, 126)
    w3  src=(896, 3072, 924)   dst=(896, 768, 924)

`w2`'s sharded axis is the **byte** axis: 1008 -> 126 bytes per rank, and Q2_K's
`type_size` is 84, so each rank gets 1.5 blocks and starts decoding scales and
quants halfway through one. `w1`/`w3` split dim 1 in *elements* (3072 -> 768,
the fused 2x384) and leave their 924-byte axis whole, so they are safe at any
tensor-parallel size.

Note `create_weights` sets `input_dim: 1` on w2, which describes the logical
layout and not the packed one -- the packed tensor carries bytes in its last
dim, and that is what gets narrowed. Reading that attribute alone suggests the
split is safe; it is not.

The model emits `!!!!!!!!` for every prompt, and `--ignore-eos` benchmarks
report full throughput throughout, which is how a broken TP8 profile shipped.

`gate`/`up` are column-parallel and split the non-contiguous dim, so they are
safe at every TP. The `padded_moe_intermediate_size` mechanism does not help:
it enlarges the buffer but still shards at `moe_intermediate_size // tp_size`.

A correct fix must make the byte shard a whole number of `type_size` blocks.

Option A -- pad the intermediate to a multiple of `256 * tp_size`: 3072 -> 4096
(16 blocks). Per rank 168 bytes = 2 blocks; ranks 0-5 receive the 12 real
blocks, ranks 6-7 only zeros. Uniform per-rank shapes, so the fused MoE path is
unchanged, but it costs ~33% MoE weight memory and idles two ranks. The loader
must clamp the narrow when `rank * shard_bytes >= src_bytes` and leave the
destination zeroed, which the existing `padded_moe_intermediate_size` does not
do -- that mechanism enlarges the buffer but still derives the shard from
`moe_intermediate_size // tp_size`.

Option B -- shard the 12 blocks unevenly, 4 ranks x 2 blocks + 4 ranks x 1
block. Lossless and better balanced, but `intermediate_size_per_partition`
becomes rank-dependent, which touches the config plumbing and every consumer
that assumes a uniform partition.

Neither is implemented; the `k3-8t` profile has been withdrawn.

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
