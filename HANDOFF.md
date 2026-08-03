# Kimi K3 GGUF on MI300X — handoff

Updated: 2026-08-03 UTC

## Status

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
    enforce_eager=True,
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

## Why TP6

TP8 is invalid for the only viable ROCm MLA backend: 96 heads / 8 ranks gives
12 heads per rank, while AITER MLA requires a head count that is a multiple or
divisor of 16. TP6 gives 16 heads per rank and fits the 858 GB checkpoint at
about 142 GiB of weights per GPU. TP4 would exceed the memory budget and also
fails the head rule. Two GPUs therefore remain idle.

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
- No duplicate-work/PR checks were run because no PR to upstream was proposed
  or opened. A human must review every changed line and run the relevant tests
  before submitting anything upstream.
