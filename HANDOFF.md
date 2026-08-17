# Qwen3.8-27B on Metal — bring-up handoff

Updated: 2026-08-17. Supersedes the A100 DSV4 handoff previously in this file
(that campaign's findings are fully recorded in perf/optimization_status.md;
the old text is in git history at `bad7cfd46`).

## Mission

Add serving support for Qwen3.8-27B to this stack on Apple Silicon (M1 Ultra,
128 GiB), measure it with the exact-token harness, and open a PR. The
deliverable is the same shape as the Muse-Glimmer and DSV4 Metal campaigns:
profile boots to health, serves the benchmark workload correctly, tok/s
recorded in the notebook against the local bars (llama.cpp and MLX on the
same box), PR from the fork.

Work happens on a branch/worktree off `qwen38-bringup` (see State below).

## What the model is (source of truth: Qwen/Qwen3.8-27B config.json, verified)

- Architecture string `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`
  (the 3.8 release reuses the 3.5-series architecture; there is NO
  Qwen3_8-specific class anywhere).
- Dense 27B VL hybrid. 64 text layers = 48 `linear_attention` (Gated
  DeltaNet) + 16 `full_attention`, pattern 3:1 (`full_attention_interval` 4).
- Gated DeltaNet layers: 16 K heads x 128, 48 V heads x 128 (3x V:K ratio),
  causal conv kernel 4, `mamba_ssm_dtype` float32, output gate swish.
  Scalar-gate GDA style — llama.cpp's Metal kernel covers exactly this
  (GDA, head_size 128).
- Full-attention layers: 24 Q / 4 KV heads, head_dim 256,
  `attn_output_gate` true, partial_rotary_factor 0.25, interleaved mrope
  (sections [11, 11, 10]), rope_theta 1e7.
- hidden 5120, intermediate 17408 (silu), vocab 248320, native context
  262144, bos=eos=248044, GemmaRMSNorm-style norms (eps 1e-6).
- Vision tower ships as a separate mmproj GGUF (deepstack ViT); text-only
  serving first, exactly the muse_glimmer pattern.
- Ships an MTP drafter (`mtp_num_hidden_layers` 1, shared embeddings) as
  separate mtp-*.gguf files — future speculation work, method `qwen3_5_mtp`.

## What already landed (branch `qwen38-bringup`, commit `b7bfb1f3d`)

- `slimserve/profiles.json`: source `qwen38-27b` (ggml-org GGUFs Q4_K_M
  17.7 GiB + Q8_0 26.6 GiB, shared mmproj, MTP speculator declared) and
  profile `qwen38-1` (metal, 32768 bring-up context, 8 GiB KV pool,
  reasoning_parser qwen3, tool_call_parser qwen3_xml, prefix caching OFF —
  linear-attention state is not block-sharable — speculation OFF).
  48/48 profile tests pass; `slimserve qwen38-1 --dry-run` resolves.
- Model downloaded and byte-verified:
  `~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` (+ mmproj Q8_0).
- Full gap analysis + milestone plan in perf/optimization_status.md
  (2026-08-17 "Qwen3.8-27B Metal bring-up" entry).
- Upstream vLLM `qwen3_5.py` fetched for study to the previous session's
  scratchpad (`upstream_qwen3_5.py`) — refetch if the scratchpad is gone:
  raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_5.py

## The four gaps (all verified in-tree — this is why it does not serve today)

1. **Model class missing.** Our registry's `qwen3_5` row is DANGLING (this
   fork stripped the module). Upstream chain to vendor and adapt:
   `qwen3_5.py` (732 lines) -> `qwen3_next.py` (~32 KB, the hybrid decoder
   this "mostly" is) + `qwen3_vl.py` (~117 KB, vision — defer) +
   `qwen2_moe.py` (only for the `Qwen2MoeMLP` class) +
   `transformers_utils/configs/{qwen3_5,qwen3_5_moe}.py`. Interfaces in this
   fork have drifted; expect adaptation, not copy-paste. Register BOTH the
   dense `Qwen3_5ForConditionalGeneration` row and fix the Moe row.
2. **GDN compute is Triton-only.** `vllm/model_executor/layers/mamba/gdn/`
   (qwen_gdn_linear_attn.py — already in-tree from Kimi K3) calls
   `vllm/third_party/flash_linear_attention/ops/*` (chunked delta rule,
   fused recurrent, cumsum, sigmoid gating — every file Triton) plus
   `mamba/ops/causal_conv1d` Triton kernels. None run on MPS.
3. **Hybrid state cache unproven on Metal.** `gdn_attn.py` backend +
   `v1/worker/gpu/model_states/mamba_hybrid.py` exist (K3 campaign,
   CUDA/ROCm-proven); their metadata builders may carry Triton or CUDA
   assumptions. Metal also needs `check_and_update_config` to accept a
   mamba/hybrid model (block_size, state dtype fp32).
4. **GGUF weight mapping.** This fork's loader (gguf_weight_utils.py) needs
   qwen3_5 tensor-name mappings for llama.cpp's arch. Two llama.cpp-side
   layout facts that WILL bite: the 3.5-series conversion "fixed the qkvzba
   order" by splitting the GDN in-proj into separate qkv/z/b/a tensors, and
   V heads were REORDERED to avoid interleaved repeats in linear attention.
   Read the conversion code in llama.cpp PR #19468 before trusting any
   tensor reshape. The loader must also ignore mmproj-* and mtp-* files.

## Reference implementations to study (in priority order)

**llama.cpp — the direct Metal reference:**
- PR #19468 "[MODEL] support qwen3.5 series" (JJJYmmm) — the model graph,
  conversion (tensor naming/ordering), pre-tokenizer, partial IMRoPE.
  Author's note: "mostly qwen3 next + imrope". Built on the Qwen3-Next PRs
  #19435 and #19456.
- **PR #19504 "ggml: add GATED_DELTA_NET op" (am17an) — a FUSED METAL
  KERNEL for the gated delta-net recurrence.** Supports GDA (scalar gate)
  and KDA (per-row gate), head_size 64 and 128 — Qwen3.8's exact shape.
  This is the single most valuable artifact for milestone M3: study the
  Metal source in `ggml/src/ggml-metal/` and adapt the algorithm (not the
  ggml plumbing) into `csrc/quixicore/metal/` per our vendoring rules.
- `docs/ops.md` in llama.cpp for the op/backend support matrix.
- Note: there is no ~/llama.cpp checkout on this Mac (CLAUDE.md's reference
  paths are for the Linux boxes) — clone it locally as a reference tree,
  and get the llama.cpp tok/s bar on this box while at it (Muse precedent:
  the notebook records "llama.cpp reference on this box: 26.6/50.2").

**MLX — the Metal-native algorithm reference:**
- mlx-lm serves Qwen3.8-27B end to end on Apple Silicon
  (mlx-community/Qwen3.8-27B-4bit and -bf16). Read mlx-lm's qwen3_5 model
  file for a clean, GPU-friendly formulation of the recurrence and its
  state cache, unencumbered by Triton. Also a numerics oracle: run the
  bf16 conversion for greedy-token comparison, and record the MLX tok/s
  bar on this box next to llama.cpp's.

**vLLM upstream — the serving-stack reference:**
- `qwen3_5.py` + `qwen3_next.py` (model graph, hybrid cache wiring,
  `mamba_block_size` handling); the Qwen3-Next blog post describes the
  hybrid design intent.
- Bugfix PR #36329 "Fix Qwen3.5 GatedDeltaNet in_proj_ba Marlin failure at
  TP>=2" — a weight-packing gotcha worth reading even though we are TP1.
- vLLM recipes page for Qwen3.5 (recommended flags, parsers).

**SGLang — secondary serving reference + cautionary tales:**
- Qwen3.5 cookbook page; issue #31594 (ROCm hang inside
  `chunk_gated_delta_rule_fwd` under kernel serialization on the
  linear-attn state path) — remember it if the Metal port hangs the same
  way; issue #31969 documents how GDN conv/recurrent state crossing request
  boundaries breaks isolation — relevant to prefix caching staying OFF.

**In-tree assets (stand on our own shoulders):**
- `vllm/model_executor/layers/mamba/gdn/{base,qwen_gdn_linear_attn,kimi_gdn_linear_attn}.py`
  — the layer is already here and CUDA/ROCm-proven via Kimi K3.
- `vllm/v1/attention/backends/gdn_attn.py` + `v1/worker/gpu/model_states/mamba_hybrid.py`.
- `vllm/model_executor/models/muse_glimmer.py` — text GGUF + separate
  mmproj, Metal dense/GQA serving, reasoning-parser wiring: the closest
  end-to-end precedent for THIS bring-up.
- `csrc/quixicore/metal/` + `tk_launch.h` + `qc_metal_serving.mm` — the
  MMVQ family (q4_K included) and Muse's multi-row `qgemv_mm` cover the
  dense projections/FFN; kernel oracle-test conventions live next to the
  existing kernels.
- `vllm/platforms/metal{,_compat}.py` — compat patches, and where hybrid
  models will need config handling.

## Milestone plan (gates between each; record every step per perf/perf.md)

- **M1 — first tokens (correct before fast).** Vendor + adapt the model
  chain, text-only. GGUF mapping (respect the qkvzba split + V reorder).
  Torch-native MPS fallbacks for: chunked/recurrent gated delta rule,
  causal_conv1d fn + update, gated RMSNorm, sigmoid gating (keep
  `mamba_ssm_dtype` float32 state). Wire gdn_attn/mamba_hybrid on Metal.
  Gate: greedy tokens match a reference (MLX bf16 or HF transformers CPU)
  on short prompts; profile boots to health and serves the primer.
- **M2 — baseline.** Exact harness (1000-in/8-out, 1000-in/2000-out, c1)
  through `slimserve qwen38-1 --serve`; pin shas + counters; notebook
  entry. Expect slow decode (eager torch recurrence) — that is fine; it is
  the baseline.
- **M3 — Metal GDN kernels.** Adapt llama.cpp PR #19504's fused Metal
  recurrence into `csrc/quixicore/metal/` (our launch/PSO conventions, not
  ggml's), plus conv1d update. Oracle tests vs the torch fallback
  (bitwise/tolerance per existing kernel-test style). A/B e2e, bit-exact
  or documented-ULP, tok/s recorded.
- **M4 — dense-path reuse + bars.** Route the attention/FFN projections
  through the existing MMVQ/qgemv_mm kernels where profiling says it pays.
  Record llama.cpp and MLX tok/s bars on this box; compare honestly.
- **M5 — stretch.** 262144 context sizing, vision via mmproj, MTP
  speculation (`qwen3_5_mtp`, drafter GGUF already published). Each is its
  own gated entry.

Then: PR from the fork (auroter has no QuixiAI write access until Eric
grants it) — `gh pr create --repo QuixiAI/SlimServe --base main --head
auroter:<branch>`. Note it stacks on PR #2 + PR #3 if those are still
unmerged. PR description: short-to-medium, user-friendly, list what was
done and where the gains are, all performance in tok/s, compare against the
llama.cpp and MLX bars on the same box.

## Ops constraints (hard-won, do not relearn)

- Commits authored `auroter <sean.gherardi@lazarus.enterprises>`. No
  co-author/assistance trailers. (Overrides the CLAUDE.md Eric-authorship
  line.)
- Explain diagnosis + proposed change to the user before code edits.
- Boot protocol (dsv4 profile, and adopt for qwen until proven otherwise):
  `pkill -f api_server`, wait `memory_pressure -Q` >70% free, boot
  `PYTHONUNBUFFERED=1 .venv/bin/slimserve <profile> --serve -y` in
  background, health = "Application startup complete" (exclude
  fa_utils/FA2 lines from error greps). Ramp before multi-chunk prefill:
  primer, then a short request with real decode, then a long decode. The
  async-output wedge is structurally fixed (PR #3) but the protocol stays
  until soak.
- `sysctl iogpu.wired_limit_mb` must be 122880 (resets on machine restart).
- Shell is zsh-eval: no bare `===`, no BSD-grep `\|` alternation, beware
  multi-line command substitutions in `[ ... ]`.
- Interval logs are diagnostics; only exact-harness output counts.
- DSV4 anchors (must stay green if shared code is touched — UPDATE 29):
  8-tok `db2846cf721b` 7/10/2; off1-2000 `7ce993786ba1` 1538/2320/464
  ~63 s; 2500x64 `e973493bef44` 51/60/12 ~3.4-3.5 s. Any flip: config A/B
  before blaming code (the 2500 anchor re-rolled environmentally once
  already; method in UPDATE 29).

## State at handoff

- Branches (all pushed to fork auroter/SlimServe):
  `metal-m1ultra-campaign` = PR #2 (CLEAN), `metal-async-output-wedge-fix`
  = PR #3 (CLEAN, stacked on #2), `qwen38-bringup` = this campaign's base
  (stacked on #3). QuixiCore-Metal PR #3 open and CLEAN.
- Box: server DOWN, memory free, worktree clean on `qwen38-bringup`.
- Model files: `~/models/Qwen3.8-27B-GGUF/{Qwen3.8-27B-Q4_K_M.gguf,mmproj-Qwen3.8-27B-Q8_0.gguf}`.
  Q8_0 text and the mtp drafter download on demand via the profile.
- Next command: branch/worktree off `qwen38-bringup`, then start M1 with
  the llama.cpp clone + upstream file study.
