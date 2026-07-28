# GLM-5.2-Vision GGUF on vLLM — work log

## STATUS: WORKING (2026-07-27 ~04:03)

Text and vision both generate correctly on TP2 / 2x MI300X, CUDA graphs on, no
instrumentation. Root cause was one line in the GGUF plugin — see "THE BUG" below.

```
'The capital of France is' -> ' Paris. The city is located on the River Seine in northern France.'
blue circle image          -> 'The image shows a blue, stylized, abstract shape...'
red triangle image         -> 'The image shows a red, three-dimensional, triangular shape...'
89-token instruction       -> correct structured summary + correct explanation of all-reduce
```

Startup 272 s. Generation ~2.8 s for 3x16 tokens.

**THE ONE THING THAT MUST NOT BE LOST** (uncommitted, deliberately — committing was
not requested):
`~/vllm-gguf-plugin/vllm_gguf_plugin/quantization/params.py` — `tp_rank` must come from
`get_tensor_model_parallel_rank()`, not `kwargs.get("tp_rank", 0)`.
vLLM-side: `vllm/utils/import_utils.py`, `vllm/model_executor/layers/mhc.py`,
`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `requirements/*`.

Numerical agreement with llama.cpp is good: **median per-element error 0.87%** across all
78 layers, worst non-final layer 3.2%. The earlier "12% drift by `l_out-60`" and "`l_out-77`
is genuinely off" claims were BOTH artifacts of my comparison method — see
"Drift and l_out-77: resolved" below. Neither is a model defect.


> Living state file for the `/loop` investigation. Read this first, every iteration.
> Update it before each iteration ends. `<SCRATCH>` =
> `/tmp/claude-1000/-home-hotaisle-vllm/b82e21e3-d7db-452c-9025-da31ee8de79a/scratchpad`

## Iteration log

- **iter 1 (2026-07-27 03:0x)** — Localized the divergence to `down_proj` by controlled
  measurement (matching input `ffn_swiglu-0`, diverging output `ffn_out-0`, all 5 tokens).
  Everything upstream of it — embedding, full attention block, gate/up/SwiGLU — proven
  correct across all token rows. Testing the TP row-shard hypothesis now.
- **iter 2** — **Row-shard hypothesis DEAD**, killed by `<SCRATCH>/test_down.py` (single
  GPU, no server). `blk.0.ffn_down.weight` is Q8_0, 13056 B/row, half = 6528 B = exactly
  192 blocks. Measured rel-err vs a float64 gguf-py dequant reference:

  | ntok | full | shard0+shard1 | shard vs full |
  |------|------|---------------|---------------|
  | 1    | .0060 | .0057 | .0040 |
  | 5    | .0039 | .0041 | .0027 |
  | 64   | .0058 | .0056 | .0031 |

  All ~0.005 = bf16+quant noise. The GGUF matmul kernel is correct for Q8_0 on both the
  MMVQ (ntok<=2) and MMQ (ntok>2) paths, and byte-half row-parallel sharding is correct.
  `down_proj` is fine in isolation. **Do not retry either.**

  This forces the real conclusion: **the probe has only ever seen rank 0.** Under TP2
  `gate`/`up` are column-parallel, so rank 0 holds intermediate cols [0:6144] and rank 1
  holds [6144:12288]. Printing 3 of 12288 columns from rank 0 cannot see a broken rank 1
  — and a broken rank-1 half is exactly consistent with "correct `down_proj` local math,
  wrong `ffn_out`", because row-parallel `down_proj` all-reduces a good local half with a
  garbage remote half. Rule #1 biting in the same place a second time.

  Probe upgraded in `deepseek_v2.py::_qf_probe`: now prints TP rank, first AND last 3
  columns, the full-tensor float64 SUM, and the shard width. Run -> `<SCRATCH>/sums1.log`.


**Goal:** serve `/home/hotaisle/models/GLM-5.2-Vision-GGUF/antirez-routed/GLM-5.2-UD-Q2_K_RoutedQ2K-00001-of-00006.gguf`
with `mmproj-GLM-5.2-Vision-f16.gguf`, TP2 on 2x MI300X, producing coherent text.

**Done when:** prompt `"The capital of France is"` yields a coherent continuation, AND
`l_out-0 … l_out-77` sums track llama.cpp's reference within a few percent.

---

## Reference (llama.cpp, ground truth)

Prompt `"The capital of France is"` -> ids `[785, 6722, 315, 9621, 374]`.
Full eval-callback dump: `<SCRATCH>/evalcb.log`; layer sums `<SCRATCH>/ref_layers.txt`.

```
embd         t0 .0198,.0044,.0005 | t1 .0103,-.0103,.0022 | t2 .0054,.0051,.0085 | t3 .0122,.0003,-.0017 | t4 .0115,.0006,.0017
ffn_inp-0    t0 .0185,.0042,.0042 | t1 .0087,-.0107,.0056 | t2 .0042,.0060,.0105 | t3 .0095,.0001,.0012 | t4 .0091,.0005,.0033
ffn_gate-0   t0 .0884,.0042,.0721 | t1 .1234,-.0035,.0425 | t2 .0653,-.0165,.0251 | t3 .0762,.0031,.0593 | t4 .0383,.0052,.0510   sum 3375.118652
ffn_up-0     t0 -.0026,.0249,.0405 | t1 -.0384,.0193,-.0173 | t2 -.0142,.0434,.0212 | t3 .0224,.0415,.0160 | t4 .0074,.0538,.0538  sum -14.332284
ffn_swiglu-0 t0 -.0001,.0001,.0015 | t1 -.0025,-.0000,-.0004 | t2 -.0005,-.0004,.0003 | t3 .0009,.0001,.0005 | t4 .0001,.0001,.0014 sum -0.577505
ffn_out-0    t0 -.0067,-.0014,-.0030 | t1 .0025,.0084,.0023 | t2 -.0018,-.0006,-.0048 | t3 -.0000,.0038,.0008 | t4 -.0006,-.0023,-.0018 sum 1.525841
l_out-0 sum 1.844878   l_out-9 sum 230.6999   l_out-23 sum 130.70
result_norm 31.0185   result_output sum 108999.8438
```

---

## Proven correct in vLLM (measured against the above, ALL 5 token rows)

- tokenizer -> ids
- `embd` — exact, all 5 tokens
- entire attention block: `ffn_inp-0` exact, all 5 tokens (so norms, q_a/q_b, kv_a/kv_b,
  RoPE, MLA absorption, o_proj are all fine)
- `ffn_gate-0`, `ffn_up-0`, `ffn_swiglu-0` — match all 5 tokens (~1e-3 rel, quant/kernel noise)

## THE BUG — FOUND (iter 3), fix applied, verification in flight

`~/vllm-gguf-plugin/vllm_gguf_plugin/quantization/params.py`, `_GGUFParamLoadMixin`:

```python
tp_rank = kwargs.get("tp_rank", 0)   # nobody passes tp_rank -> always 0
```

vLLM's `MergedColumnParallelLinear.weight_loader_v2` (`linear.py:988`) passes only
`shard_id`, `shard_offset`, `shard_size` — **never `tp_rank`**. So every rank narrowed to
rows `[0:shard_size]` and **both ranks loaded the identical `gate`/`up` shard**. vLLM's own
`BasevLLMParameter` uses `self.tp_rank`, set from `get_tensor_model_parallel_rank()`
(`parameter.py:65`); the plugin's mixin overrode that with a kwarg lookup that silently
defaulted to 0. `load_column_parallel_weight`/`load_row_parallel_weight` in the same mixin
call `get_tensor_model_parallel_rank()` correctly — only the *merged* and *qkv* paths were
wrong, which is why plain column-parallel and row-parallel tensors were all fine.

**Proof (`<SCRATCH>/sums3.log`, eager, TP2, full-tensor float64 sums):**

```
r0 ffn_gate-0 SUM 1681.035858 width 6144
r1 ffn_gate-0 SUM 1681.035858 width 6144   <- bit-identical; column-parallel halves
r0 ffn_up-0   SUM  -10.594030                 CANNOT be equal unless both ranks
r1 ffn_up-0   SUM  -10.594030                 hold the same weights
```

Mechanism of the observed `ffn_out-0` divergence: `down_proj` is sharded *correctly*, so
rank 0 computes `W_down[:, 0:6144] @ swiglu[0:6144]` (right) while rank 1 computes
`W_down[:, 6144:12288] @ swiglu[0:6144]` (correct weights, wrong activations). The
all-reduce sums one correct half with one garbage half — matching input, diverging output.
Intermediate columns `[6144:12288]` were never computed at all.

`load_qkv_weight` had a second, independent bug: it special-cased only `("k", "v")` for the
replication divide, leaving `index_k`/`index_q` unscaled. vLLM divides for everything that
is not `"q"`, which is how `index_k` (`num_heads == tp_size`, see `linear.py:1560`) lands
fully replicated. Both are fixed to mirror `BasevLLMParameter`.

### VERIFIED (`<SCRATCH>/fix1.log`) — the model generates correctly

```
'The capital of France is' -> ' Paris. The city is located on the River Seine in northern France. Paris is'
'Once upon a time'         -> ', there was a little girl named Lily who loved to play with her dolls.'
'2 + 2 ='                  -> ' 5\nI have been thinking about the phrase "2 + 2 ='
```

Gate/up sums now differ across ranks and add to llama's, as predicted:

| tensor | r0 | r1 | r0+r1 | llama | rel |
|---|---|---|---|---|---|
| `ffn_up-0` | -10.594030 | -3.736383 | -14.330413 | -14.332284 | 0.01% |
| `ffn_swiglu-0` | -0.797891 | +0.224085 | -0.573806 | -0.577505 | 0.6% |
| `ffn_out-0` | 1.524732 | 1.524732 (all-reduced) | | 1.525841 | 0.07% |

All 78 `l_out-N` prefill sums vs llama: `l_out-0` 0.17%, `l_out-9` 0.64%, `l_out-23` 0.95%
(was sign-flipped -90.94), `l_out-40` 5.0%, `l_out-60` 12.1%; median 3.82%.

**Open, not chased:** drift grows with depth (Q2_K experts + bf16 vs llama f32), and
`l_out-77` is llama -5.6705 vs vLLM -963.86 — a real absolute gap next to `l_out-76`'s
~1830, not merely a small-denominator artifact. Layers 7/8 large rel% ARE small-denominator
artifacts (llama -0.41, 3.85). Revisit if output quality disappoints; check whether layer 77
is the MTP/nextn layer and whether llama.cpp treats it differently.

**When comparing `l_out-N`, filter to `shape=(5, 6144)` and take the FIRST occurrence per
layer.** `_qf_lout` has no one-shot guard, so it also fires on all 16 decode steps; a naive
`sort -u` mixes decode rows in and produces nonsense (first attempt showed 117460% error).

## Superseded — the divergence as originally observed

`ffn_out-0` = `down_proj(swiglu)` diverges for **every** token, with a **matching input**:

```
tok0  llama -.0067,-.0014,-.0030   vLLM -.0043,-.0029,-.0043
tok1  llama  .0025, .0084, .0023   vLLM  .0005, .0050, .0014
tok2  llama -.0018,-.0006,-.0048   vLLM -.0017,-.0004,-.0021
tok3  llama -.0000, .0038, .0008   vLLM -.0022, .0011,-.0022
tok4  llama -.0006,-.0023,-.0018   vLLM -.0023,-.0010,-.0004
```

**Leading hypothesis — TP row-sharding of a GGUF tensor along the input dim.**
`down_proj` is the only MLP matrix that is `RowParallelLinear`. `gate`/`up` are
ColumnParallel: sharded on the *output* dim, which in GGML's row-major layout is a
contiguous byte range, so a naive byte-space split is accidentally correct.
`blk.N.ffn_down.weight` is `{ne0=12288 in, ne1=6144 out}` and must be split on the
*input* dim — a **strided slice inside every row** (blocks 0..191 vs 192..383 of each
384-block row for Q8_0), NOT a contiguous byte range. A contiguous split hands each rank
the wrong half and the all-reduce sums garbage.

This also explains why attention passes: MLA's row-parallel `o_proj` is not GGUF-sharded
on that axis in this model.

**Next action:** CPU-only test, no GPU, no server. Load `blk.0.ffn_down.weight` from the
GGUF, dequantize the full tensor, and compare what vLLM's weight_loader hands rank0/rank1
against the correct strided input-dim split. Look at
`vllm/model_executor/layers/quantization/gguf.py` (`GGUFLinearMethod`, its
`weight_loader`/param packing) and `RowParallelLinear.weight_loader` in
`vllm/model_executor/layers/linear.py`. The plugin (`~/vllm-gguf-plugin`) does not shard;
vLLM core does.

If that test exonerates sharding, the next suspects are the GGUF MMVQ/MMQ kernel for the
`down` quant type (check what quant `blk.0.ffn_down.weight` actually is) and the
all-reduce itself.

---

## Drift and l_out-77: resolved (both were measurement artifacts)

**1. A SUM is a bad similarity metric — cancellation amplifies relative error ~10x.**
Comparing per-element (first3/last3 from `ref_vals.txt` vs the probe's `head=`/`tail=`)
instead of by sum:

| metric | by sum | per element |
|---|---|---|
| `l_out-60` | 12.1% | **1.2%** |
| `l_out-40` | 5.0% | 0.6% |
| median, 78 layers | 3.82% | **0.87%** |

Worst non-final layer is `l_out-70` at 3.2%. Deep-layer values in vLLM land on coarse
steps (13.125, 17.625, 23.75) — bf16 mantissa spacing at magnitude ~16 vs llama's f32.
That fully accounts for the 1-3% deep-layer difference. Benign.

**2. `l_out-77` was never a divergence — llama.cpp computes only the final position in
the last layer.** `ref_layers.txt` shows `l_out-77` has shape `6144, 1, 1, 1` while every
other layer is `6144, 5, 1, 1`. So llama's single row IS token 4, and I was comparing it
against vLLM's token 0 (946% "error") and its 1-row sum -5.67 against vLLM's 5-row sum
-963.86. Different quantities in both cases. The shape column was in the reference file I
had already read.

**Rule: when comparing against a llama.cpp eval-callback dump, always check the shape
column first.** Only compare like-for-like rows.

## Rules learned the hard way — do not repeat

1. **Never declare a tensor "identical" from 3 elements of row 0.** Compare every token
   row, and compare the full-tensor SUM. That mistake cost ~12 hours; the sums disagreed
   the whole time while row 0 matched.
2. **Only trust a controlled measurement.** Matching input + diverging output on one op is
   proof. Anything else is a hypothesis, and hypotheses get tested, not announced.
3. **Never "fix" a kernel without first checking the invariant it relies on.** Changing 20
   `block_num_y` sites in `moe.cuh` to ceiling division broke it — `moe_align_block_size`
   already pads to a multiple of block_size, so the extra block read out of bounds.
   Reverted.
4. **`pgrep -f "..."` matches its own wrapper shell.** `until ! pgrep -f "pip install"`
   never exits. Cost ~25 min, three times. Use `pgrep -f ... | grep -v $$` or a PID file.
5. Reinstalling the plugin with plain `pip install -e .` re-resolves vLLM and reinstalls
   tilelang. Always `--no-deps`.
6. **Probing requires `enforce_eager=True`. There is no way to probe the compiled
   path.** Probes call `.tolist()`/`.item()` on float tensors, which dynamo cannot
   trace (`Unsupported: Tensor.tolist() with non-integer tensor`).
   `@torch._dynamo.disable` does NOT rescue it — vLLM compiles fullgraph, so a
   disabled callee is itself fatal (`Unsupported: Skip calling
   torch.compiler.disable()'d function`). Both were tried and both killed the
   workers at startup. `serve_glm.py` now sets `enforce_eager=_QF_DUMP`, so
   `QF_DUMP_LAYERS=1` implies eager. The `@torch._dynamo.disable` decorators on
   `_qf_probe`/`_qf_lout` (`deepseek_v2.py`) and `_qf` (`mla.py`) are harmless
   no-ops in eager; they must be removed with the rest of the instrumentation.
7. `Bash(run_in_background=true)` + a trailing `&` inside the command double-forks: the
   tool reports exit 0 instantly while the real process runs on. Never use both.

## Landed fixes — keep these, do not regress

- `vllm/model_executor/layers/mhc.py` — `_has_tilelang_mhc()` returns False on gfx942
  **before** probing. Root cause: importing tilelang loads `libhip_stub.so`, which exports
  `hipGetDevicePropertiesR0600` backed by the legacy R0000 layout; it interposes on every
  later HIP call, so aiter reads `warpSize` at offset 308 = `clockRate` = 2100000.
  Proven by a 30 s single-variable repro. tilelang 0.1.12 has the identical bug.
- tilelang removed from `requirements/rocm.txt`, `requirements/test/rocm.in`,
  `requirements/test/rocm.txt`.
- `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` — `hf_text_config.index_topk`
  (not `hf_config`); added and correctly populated `num_decodes`, `num_prefills`,
  `num_decode_tokens`, `prefill_max_seq_len`, `prefill` on `ROCMAiterMLASparseMetadata`.
- `~/vllm-gguf-plugin/vllm_gguf_plugin/weight_utils.py` — zero-copy `_as_tensor` via
  `torch.from_numpy` (weight load 271 s -> 102 s). Pinned staging was tried; it is slower.
- Startup 730 s -> 266 s (also: graph capture 51 -> 4 sizes, warm AOT cache). Verified
  numerically neutral by byte-identical logprobs.

## Debt

- [x] **Instrumentation removed.** `git checkout` of `deepseek_v2.py` + `mla.py`; both
      diffs were pure debug (verified line by line first — the `QF_EXACT` `q_a_proj`
      branch and the `MLAModules.q_a_proj` field were the only non-probe additions and
      both were debug scaffolding). 0 `_qf`/`_QF` references remain. To re-instrument,
      re-add probes AND remember rule 6 (probe ⇒ `enforce_eager`).
- [x] **Durable tilelang guard**, moved to the real chokepoint: `has_tilelang()` in
      `vllm/utils/import_utils.py` now returns False on gfx942 *before* `_has_module`
      does its trial import. Both consumers (`mhc.py`, `kernels/mhc/tilelang_kernels.py`)
      route through it, so no lazy import site can slip past. Verified:
      `has_tilelang() = False` and `'tilelang' not in sys.modules`. The narrower
      `mhc.py::_has_tilelang_mhc` guard is left in place as defence in depth.
- [ ] **Vision path.** `detect_gguf_multimodal` globs only `model_path.parent`
      (`gguf_utils.py:186`), i.e. `antirez-routed/`, but the mmproj lives one level up in
      `GLM-5.2-Vision-GGUF/` — so it returned `None` and vision was **silently** off, with
      no warning. Symlinked `antirez-routed/mmproj-GLM-5.2-Vision-f16.gguf -> ../` and
      confirmed detection now succeeds. NOT yet run end to end: `serve_glm.py` still sets
      `limit_mm_per_prompt={"vision_chunk": 0}`. Test harness is
      `<SCRATCH>/serve_vision.py`. **Run 1 (`vision1.log`) got the engine up WITH the
      vision tower** — startup 663 s vs 272 s text-only, so the mmproj really is found and
      loaded — then failed in the harness, not the model:
      `TypeError: object of type 'Image' has no len()`. `multi_modal_data` wants a LIST:
      `{"vision_chunk": [img]}`, not a bare PIL image. Fixed; run 2 -> `vision2.log`.
      Modality key is `vision_chunk` (inherited from Kimi-K2.5, `kimi_k25.py:176`);
      placeholder token is `<|image|>` (id 154854, `glm5v.py:54`).

      **Correct `multi_modal_data` shape** (run 2 `'Image' object is not subscriptable`
      proved a bare list is also wrong): a vision_chunk item is a TypedDict, not an image.

      ```python
      from vllm.multimodal.inputs import VisionChunkImage
      {"vision_chunk": [VisionChunkImage(type="image", image=pil_img, uuid=None)]}
      ```

      Video form is `VisionChunkVideo(type="video_chunk", video_chunk=[frames])`.
      Authoritative source is `KimiK25DummyInputsBuilder.get_dummy_mm_items`
      (`kimi_k25.py:186`) and `vllm/multimodal/inputs.py:96` — read those before guessing.
      Run 3 -> `vision3.log`.

      Vision startup: run 1 663 s (cold cache), run 2 273 s (warm). Same as text-only —
      vision costs nothing extra at startup.

      **Run 3 (`vision3.log`): exit 0, no `hipError`, but degenerate output.**
      blue circle -> `' blue blue blue blue...'`; red triangle -> whitespace;
      83-token prompt -> echoes the instruction forever. The "blue" repetition means the
      tower IS feeding real image information through (it got the colour right).
      Cause was my prompting, not the model: **this is an instruct model and I sent raw
      completion prompts.** At temp 0 that yields exactly this echo. The short text
      prompts worked only because "The capital of France is" is a natural continuation.

      **Always render through `chat_template.jinja`:**
      `tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
      enable_thinking=False)`. The template emits `[gMASK]<sop>`, `<|user|>`/`<|assistant|>`
      turns, and wraps images as `<|begin_of_image|><|image|><|end_of_image|>` — a bare
      `<|image|>` is wrong. `enable_thinking` defaults True and appends `<think>`.

      **Run 4 (`vision4.log`): VISION WORKS.** Rendered prompt verified as
      `'[gMASK]<sop><|user|><|begin_of_image|><|image|><|end_of_image|>Describe...<|assistant|><think></think>'`

      | image | output |
      |---|---|
      | blue circle | "a blue, stylized, abstract shape resembling a stylized letter M..." |
      | red triangle | "a red, three-dimensional, **triangular** shape with a hollow center" |

      Both colours correct and the triangle identified correctly — that is real visual
      grounding, not a language prior. The circle's shape was misdescribed; flat synthetic
      geometry is off-distribution and the LM is 2-bit. Worth re-checking on real photos
      before promising shape fidelity to users.

- [x] **`moe.cuh` `block_num_y`: no failure observed.** An 83-token prompt (well past the
      64-token block size) ran clean — no `hipErrorInvalidValue`, no garbage, sane
      throughput. The degenerate text was prompt formatting, not the kernel. Not a
      exhaustive proof for very long sequences, but the specific concern did not
      reproduce; treat the earlier unverified "it IS a real bug" note as withdrawn.
- [ ] Re-verify the `moe.cuh` `block_num_y` question independently with a >64-token prompt.
      Empirical test (long prompt -> coherent output, no `hipErrorInvalidValue`) is cheaper
      and more trustworthy than re-reading the kernel; fold it into the next run. Note the
      earlier "it IS a real bug for that path" claim was never verified — treat as unknown,
      not as known-bad.
- [ ] `l_out-77` gap (see above) — revisit only if output quality disappoints.
- [ ] Plugin fix in `~/vllm-gguf-plugin/vllm_gguf_plugin/quantization/params.py` is
      uncommitted. **Left uncommitted deliberately** — committing was not requested. This
      is the single change that makes the model work; do not lose it. Same for the vLLM-side
      edits (`import_utils.py`, `mhc.py`, `rocm_aiter_mla_sparse.py`, requirements).

## Production config VERIFIED (`<SCRATCH>/prod1.log`)

CUDA graphs ON, zero instrumentation, TP2. Startup 272 s (24.9 s imports, ~247 s load +
capture), generation 2.8 s.

```
'The capital of France is' -> ' Paris. The city is located on the River Seine in northern France. Paris is'
'Once upon a time'         -> ', there was a little girl named Lily...'
```
