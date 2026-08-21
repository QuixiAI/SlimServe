# Qwen3.8-27B + DFlash 2 Metal Serving Handoff

Updated: 2026-08-19 (M5 Max MacBook Pro, 128 GB, ~460 GB/s measured stream)

MISSION (user, 2026-08-19): support the `qwen38-q2kxl-1` profile on Metal --
Qwen3.8-27B (unsloth UD-Q2_K_XL GGUF + mmproj-F16 vision tower) speculated
by the DFlash 2 drafter (Inco AI, released 2026-08-18;
z-lab/Qwen3.8-27B-DFlash2-GGUF Q4_K_M). Vendor bar for this exact pairing:
mean acceptance 4.80 at block 8 and 2.7-3.4x over autoregressive at batch 1
(SGLang, model card). Output must be provably identical to plain decode
(rejection sampling unchanged).

Prior campaigns, both preserved in git history:

- Muse-Glimmer-30B Metal campaign: committed at `ad8e8e937` (20.1 tok/s
  spec rested; fused verify, tensor-ops kernels, long-context fix). Its
  handoff text is in that commit's HANDOFF.md.
- DeepSeek V4 0731 A100 campaign: lives on the A100 box in
  `/home/ubuntu/SlimServe`; latest handoff at `bad7cfd46`.

## What exists already (2026-08-19, all uncommitted)

- `slimserve/profiles.json`: `sources["qwen38-27b"]` (q2kxl quant
  9,828,981,664 B + sha256, shared mmproj-F16 927,607,488 B, z-lab
  DFlash2 speculator 1,143,006,752 B + sha256) and
  `profiles["qwen38-q2kxl-1"]` (metal-only, gated `"status":
  "in-progress"`; engine mirrors muse-kdyn-1 plus `reasoning_parser:
  qwen3`, `tool_call_parser: qwen3_xml` (needs live validation),
  `enable_prefix_caching: false`, `num_speculative_tokens: 7`). Tests
  updated (48 pass). ALL THREE ARTIFACTS ARE DOWNLOADED AND
  SHA-VALIDATED in `~/models/`; both GGUFs + the mmproj are dumped and
  the ground truth lives in `perf/qwen38_metal_design.md` (READ IT --
  incl. the quant-matrix finding: 9 i-quant formats the Metal kernels
  do not cover yet). `slimserve/fetch.py` gained short-read resume and
  an already-complete-.part fast path; registry bytes/sha must come
  from `curl -I`/paths-info, never summarized pages (notebook (7)).
- Vendored target text chain (dense-trimmed): `models/qwen3_next.py`,
  `models/qwen3_5.py`, `configs/qwen3_5.py`, `configs/qwen3_next.py`,
  `layers/mamba/mamba_mixer2.py` (loader shim). Imports + registry
  resolution verified on this machine; GDN runtime kernels are still
  Triton-only (the critical path).
- GGUF plumbing VERIFIED against the real files:
  `transformers_utils/gguf_qwen35.py` (target config, DFlash2 drafter
  config, qwen35 tokenizer incl. the vendored qwen35 pre-split in
  gguf_native.py), config-parser + tokenizer-registry + gguf_loader
  three-way `dflash` dispatch on `dflash.selector_rank`.
- DFlash 2 drafter COMPLETE at unit level: `models/qwen3_dflash2.py`
  (two-tap convs -- exact vs naive reference; greedy selector walk --
  matches manual reference; Metal ctx-KV precompute borrowed from Muse;
  registered), `gguf_adapters/qwen3_dflash2.py` (81/81 tensors mapped,
  Q4_K conv/selector tensors dequantized at load), and the speculator's
  selector branch in `spec_decode/dflash/speculator.py::_generate_draft`
  (greedy only; probabilistic stays on the gumbel path). NOT yet
  load-tested end to end -- the drafter shares the target's
  embed/lm_head, so that needs the target model.

## Target model facts (unsloth config.json; verify against GGUF metadata
## once downloaded)

Qwen3_5ForConditionalGeneration (model_type `qwen3_5`) -- NOT dense qwen3:

- 64 text layers, 3:1 hybrid interleave: 48 `linear_attention` layers
  (gated-deltanet style: 16 K-heads / 48 V-heads at 128 dim, conv kernel 4,
  mamba_ssm_dtype float32) and 16 `full_attention` layers (GQA 24/4,
  head_dim 256, partial_rotary_factor 0.25, interleaved MRoPE
  [11, 11, 10], rope_theta 1e7).
- hidden 5120, ffn 17408, vocab 248320, 262144 max ctx, attn_output_gate
  true, final logits NOT softcapped (unlike Muse).
- Native MTP head (1 layer) exists in the checkpoint -- unused by us
  (DFlash 2 replaces it; do not load it).
- Vision tower `qwen3_5_vision`: 27 blocks, hidden 1152, 16 heads, patch
  16, temporal patch 2, spatial merge 2x2, out_hidden 5120,
  num_position_embeddings 2304, gelu_pytorch_tanh. image_token_id 248056,
  vision_start/end 248053/248054.

## Drafter facts (incoai/Qwen3.8-27B-DFlash2 config.json)

- `DFlash2DraftModel`, qwen3-classed backbone: 5 layers, SWA 2048, heads
  32/8 at head_dim 128, hidden 5120 (== target), ffn 17408, vocab 248320.
- Reads target_layer_ids [5, 19, 33, 47, 61] (0-based HF numbering).
- dflash_config: block_size 8, conv_kernel_size 2, conv_group_size 16,
  selector_rank 256, selector_top_k 16, mask_token_id 248070.
- DFlash 2 = DFlash + (a) path selector: keep top-16 candidates per
  position, score adjacent pairs S_t(a,b) = U_t(b) + <A(a) o H(h_t), B(b)>
  (rank-256 bilinear, ~2M params), one greedy/sampled walk from the last
  verified token; (b) two-tap dynamic depthwise convs before AND after
  every attention/MLP sublayer (k_t0*x_t + k_t1*x_{t-1}; learned base
  kernel + per-16-channel content correction; position 0 taps the last
  verified token). Blog: inco.ai/blog/dflash2 (2026-08-18).
- Upstream vLLM runs it as method "dflash" + num_speculative_tokens 7 --
  DFlash2-ness is detected from the drafter architecture, NOT a new method
  string. Keep it that way here (pydantic `extra="forbid"` on
  SpeculativeConfig makes new method strings expensive; see
  vllm/config/speculative.py:60-76).
- llama.cpp support is an UNMERGED PR (master has DFlash 1 only); oMLX has
  it. The blog post's formulas are the spec; no local reference
  implementation yet.

## Support-surface map (scouted 2026-08-19; notebook (2) has the detail)

REUSABLE AS-IS: the whole slimserve resolve/gate/fetch stack (zero code
changes; profiles.json is the only input); the DFlash runtime
(DFlashSpeculator one-block layout `1 + num_speculative_tokens` rows,
prepare_dflash_inputs incl. the native Metal op, load_dflash_model
embed/lm_head sharing, rejection sampler); `qwen3_dflash.py`'s
DFlashQwen3Model/ForCausalLM as the drafter base (Muse subclasses it and
only overrides the two Metal-hostile methods -- do the same);
`qwen3.py::Qwen3ForCausalLM` already implements the aux-hidden-state
contract but is text-only dense -- the LANGUAGE-MODEL half of the target
does NOT exist for qwen3_5.

STATUS ROLL-UP (2026-08-20 evening -- supersedes the staged plan below
where they disagree; notebook (19)-(26) has the evidence):

- DONE since the 39efaa7d9 commit: GDN spec-state rollback (verified
  9.5e-7 incl. cross-call resume); sampled selector walk ENGAGED
  (sparse 16-way distributions -> draft_logits); DFlash 2 e2e UP --
  essay acceptance 2.71/0.244 BEATS the llama.cpp dflash2-pr oracle
  (2.51/0.219) on the same GGUFs+settings (vendor 4.80 is a GSM8K
  number; acceptance is task-domain-dependent, bench arms must match);
  VISION landed (tower parity ~1e-3 vs llama-mtmd-cli, real-image e2e
  smoke); native IQ decode for IQ2_S/IQ3_S/IQ1_M (_DEQUANT_TYPES empty,
  residency ~12 GiB); muse regression gate CLOSED (byte-identical
  seeded output; smokes must use PROFILE-EXACT kwargs).
- Numbers (OLD metallib, V2 runner, seeded shipped defaults): plain
  6.4 tok/s (V1: 2.5), spec 3.9-4.1. SPEC < PLAIN = bug per policy,
  attributed: 8-position python GDN verify scan + CPU rejection
  fallback -> the fusion workstream's first target.
- THE ONE OPEN BUG: V2-on-Metal boot-lottery decode corruption
  (prefill exact; some boots clean, others collapse ~token 20 =
  first-KV-block append; explains repetition/"!"-floods/rejection-NaN
  alike -- new IQ kernels exonerated at fp16+bf16). Hunt agent running
  with the block-append staged-write hypothesis ranked first; fix bar
  >= 6 clean boots + muse smoke.
- Also open: MPS seed-keyed sampling (spec-vs-plain seeded identity
  untestable; gumbel/rejection fallbacks unkeyed); MPS rejection
  fallback lacks _sanitize_nan parity; V2 must be forced via
  VLLM_USE_V2_MODEL_RUNNER=1 for the hybrid target (make it the
  default once the corruption is fixed); stray-token quirk at temp 1.0
  in both text+vision answers (tracked, unattributed).
- Ops gotchas: rm-then-cp + codesign -f -s - for metallib/.so
  refreshes; regression smokes use profile-exact engine kwargs.

STAGED PLAN (updated 2026-08-20: TEXT CORRECTNESS ACHIEVED):

1. ~~GGUF ground truth~~ DONE (notebook (3), (5), (7)).
2. ~~Target text model on Metal~~ **WORKING AND LAYER-PARITY-VERIFIED**
   (notebook (8)-(18)): qwen35 adapter (851/851 tensors; fused-group
   dequant; row-split GDN qkv into scalar shards), torch-native MPS GDN
   core (verified 1.4e-6 vs a numpy port of the Triton kernels; spec
   masks NotImplementedError until step 5b), METAL_MMQ routing (tile
   kernels for IQ1_S/IQ2_XS/IQ2_XXS/IQ3_XXS/IQ4_XS verified on real
   tensors), load-time dequant only for IQ1_M/IQ2_S/IQ3_S (+12 GiB).
   TWO CONVERTER CONVENTIONS were the correctness bugs (notebook (18)):
   the +1 norm fold (undone at load, _degemma_gguf_norms) and the GDN
   tiled V-head order (pairing i_k = i_hv % 16, cfg flag
   gdn_tiled_v_head_layout; FLA Triton kernels still assume grouped --
   matters if this GGUF ever runs on CUDA). All 64 layers cos >= 0.9997
   vs llama.cpp eval-callback; greedy output oracle-matching.
   REMAINING (perf, not correctness): sequential-python GDN scan ->
   chunked/kernel; head_dim-256 paged fast path; native IQ1_M/IQ2_S/
   IQ3_S decode. The env-gated _Qwen38DumpState diagnostic in
   qwen3_5.py is RETAINED intentionally (zero-cost unset; it is the
   layer-parity instrument and the spec hunt may need it) -- remove when
   the campaign closes.
3. **Vision tower**: Muse pattern; full mmproj map in the design doc.
   Text-first; config builder flips to ForConditionalGeneration when
   the classes exist.
4. ~~Drafter~~ DONE at unit level.
5. **Speculation** (next big lift, order matters):
   (a) drafter e2e load + token-identical greedy check (needs target =
   now unblocked; shares target embed/lm_head);
   (b) GDN spec-state rollback in the MPS scan (design written in
   perf/qwen38_metal_design.md: per-position state slots +
   num_accepted_tokens resume; replaces the NotImplementedError);
   (c) e2e spec: acceptance vs vendor 4.80, spec > plain (always-on
   mandate).
6. **Serve + validate**: un-gate the profile ONLY after: health, text +
   image smoke, acceptance ~4.8, greedy parity, recorded TPS with raw
   artifacts, notebook + baselines updated. First e2e milestone can be
   TEXT-ONLY with the profile still gated.
7. Reference bars (perf/baseline_status.md): llama.cpp plain 35.67
   tok/s this box/artifact; vendor DFlash 2 multiplier 2.7-3.4x =>
   ~90-120 tok/s spec target band.

## Non-negotiables carried from prior campaigns

- Speculative decoding is ALWAYS ON and must be net-positive
  (memory/spec-always-fastest: slower-than-plain spec is a bug, never a
  documented config).
- Bench discipline: rested 45-min protocol for route flips; cached-prompt
  repeats for decode numbers; interval logs are diagnostics, not
  baselines. Thermal decline within ~90 s -- compare matched positions.
- kv_cache_dtype auto (fork defaults fp8; Metal has no fp8 KV path).
- Every experiment goes in `perf/optimization_status.md`; raw logs under
  `perf/results/YYYY-MM-DD/<run-id>/`.

## Build / bench / validate

- Build (metallib + host): `cmake --build build/temp.macosx-11.0-arm64-cpython-312`
  then copy `quixicore_metal.metallib` and `_quixicore_C.cpython-312-darwin.so`
  from that build dir into `vllm/`.
- Serve (once un-gated): `.venv/bin/slimserve qwen38-q2kxl-1 --serve -y`.
- Profile tests: `.venv/bin/python -m pytest tests/slimserve/test_profiles.py -q`.
- GGUF metadata dump: `.venv/bin/python -c "from gguf import GGUFReader; ..."`
  or `vllm/transformers_utils/gguf_native.py` helpers.
