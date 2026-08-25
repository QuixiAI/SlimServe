# Muse-qwen38: single-command-buffer decode step (N14)

Decision record: user GO 2026-08-24 (trajectory re-pin accepted; muse_glimmer
precedent). Bringup is opt-in-gated — canonical stays bit-exact at
467b35c3 / d0e07ddd until the flip, which requires full quality
revalidation (262k needle, long decodes, drafter acceptance before/after)
and NEW pins.

## Why

UPDATE 23 census: ~73% of channel time sits in command-buffer-granularity
footprints. The encode-crumb tier (U43–U49, ~5 µs × dispatch count) is
exhausted at ~2% total; what remains is the CB *boundaries* themselves —
commit/schedule gaps where the GPU idles between hundreds of small command
buffers, and host encode serialized against GPU execution. The
muse_glimmer precedent (`muse_step_init/layer/run` in qc_metal_serving.mm,
`serving_glue/muse_step.metal`) collapses the whole target forward into
one `encode("muse_step", ...)` closure: weights registered once, only
per-step tensors passed per call.

## Scope (V1)

- **Eligible step**: steady uniform all-spec decode (the U43
  `steady_signature` hit path — metadata cached, shapes static),
  `m = num_reqs * (1 + k) <= 8` total rows (covers c1 k=3 → m=4; c2),
  bf16 activations, text-only (mRoPE rows identical → flat T row).
- **Inside the CB**: all 64 decoder layers (48 GDN + 16 gated full-attn)
  plus the final gemma-norm.
- **Stays eager**: embedding lookup, lm_head/sampling/verify, the DFlash2
  drafter (U47: drafter math is NOT reduction-order-tolerant — do not
  touch), all prefill and non-steady steps.
- **Env**: `VLLM_QC_MUSE` = `0` (off, default during bringup) / `1`
  (serve muse) / `shadow` (run muse AND eager, compare per-layer, serve
  eager — the bringup instrument). `VLLM_QC_MUSE_LAYERS=N` caps muse to
  the first N layers (prefix capture, eager continues from layer N) for
  divergence bisection.

## Numerics stance

Bit-exactness vs the eager path is unattainable by construction (U48:
torch-MPS eager elementwise numerics are size-dependent) and is NOT the
gate. Bringup gates are: (1) shadow-mode per-layer cos/max-ulp stats at
serving shapes; (2) trajectory quality — needle 262k, long-decode reads,
acceptance-rate delta vs canonical (U47 showed the 16-way candidate
selector amplifies ULP noise; a bad acceptance roll can eat the win);
(3) DSV4 anchors re-gate (expected to hold — their compute untouched).
Then NEW pins. Corollary: the U48 `qk_norm_rope_gate` kernel, rejected
only for forking the trajectory, is FREE to use inside muse.

## Per-step inputs (everything else is registered once)

- `x` [m, 5120] bf16 — post-embedding hidden rows.
- `positions` int64 [m] (flat T row).
- Full-attn: `block_table` int32 (×2 at emit for the page-local layout),
  `seq_lens_gpu` int32 [reqs], `slot_mapping`, `seq_lens_cpu_max` (host
  int; sizes the D=256 split-K partitions at encode time — recomputed on
  host each step, cheap).
- GDN (from the U43-refreshed `_mps_spec_cache` on the shared metadata):
  `spec_cu`, `conv_slots`, `slot_table`, `num_accepted`.

## Registration surface (.mm)

- `muse_q38_init(num_layers, hidden, heads, kv_heads, head_dim,
  rotary_dim, gdn geometry (Hk, Hv, Dk, Dv), inter, eps, max_rows, ref)`
  — allocates persistent scratch sized max_rows (h, qkv, q/k/v, gdn
  q/k/v/decay/beta, attn_out, gate, g/u/mid, per-layer residual ping-pong).
- `muse_q38_layer_common(idx, input_norm_w, post_attn_norm_w,
  gate_up(fmt, scales), down(fmt, scales))`
- `muse_q38_layer_gdn(idx, qkvz(fmt, scales), ba(fmt, scales), conv_w,
  A_log, dt_bias, gated_norm_w, out(fmt, scales), conv_state, ssm_state,
  q_scale, k_scale)`
- `muse_q38_layer_attn(idx, qkv(fmt, scales), q_norm_w, k_norm_w,
  cos_sin_cache, o(fmt, scales), dense_kv)`
- `fmt` ∈ {fp8ch, nvfp4_planar}; per-projection, matching what the CT
  linear layers loaded (unsloth dynamic mixes formats per layer).
- `muse_q38_run(x, positions, block_table, seq_lens_gpu, slots_attn,
  spec_cu, conv_slots, slot_table, num_accepted, seq_lens_cpu_max,
  num_layers_cap)` → in-place on x (final normed hidden written back).

## Emit sequence (one encode() closure)

Residual pattern: `residual` and the layer input share the fused
add-norm, exactly as the eager decoder loop:

- Layer 0: `gemma_rms_norm_dyn(x) -> h`, `residual = x`.
- Layers 1..63 entry + final: `gemma_rms_norm_add_dyn(prev_out, residual)
  -> (h, residual)`.

**GDN layer** (48×):
1. `qgemv[fmt](h) -> qkvz` [m, qkv_sz + z_sz]  (z = strided tail view)
2. `qgemv[fmt](h) -> ba`
3. `launch_gdn_fused_prepare` (spec/conv-rewind mode) → q,k,v,decay,beta
4. `launch_gdn_recur_spec` → core
5. `launch_gdn_gated_rmsnorm_f32(core, z_view)` → gated
6. `qgemv[fmt](gated) -> attn_hidden`

**Full-attn layer** (16×):
1. `qgemv[fmt](h) -> qkv` [m, 2*q_sz + 2*kv_sz]  (gate interleaved)
2. `launch_qk_norm_rope_gate` → q, gate, k  (v = strided view of qkv)
3. `launch_kv_cache_scatter(k, v, slots, dense_kv, block_mult=2)`
4. glue `muse_expand_meta`: expanded_seq_lens [reqs*q_len] from
   seq_lens_gpu (+ arange offsets) — persistent buffer, tiny dispatch.
   Expanded block table: persistent `bt2x` buffer refreshed by glue copy
   (rows = base row r/q_len, entries ×2), sized [reqs*q_len, max_blocks].
5. `launch_paged_attention` (or `_partition` + reduce when
   seq_lens_cpu_max crosses the split-K bound — mirror the bound op's
   encode-time choice) over dense_kv / dense_kv[1:].
6. glue `muse_sigmoid_mul(attn_out, gate)` (exists in muse_step.metal)
7. `qgemv[fmt](attn_out) -> attn_hidden`

**MLP** (64×):
1. `gemma_rms_norm_add_dyn(attn_hidden, residual) -> (h2, residual)`
2. `qgemv[fmt](h2) -> gate_up` [m, 2*inter]
3. `launch_qc_swiglu` (or muse_silu_mul on halves) → mid
4. `qgemv[fmt](mid) -> mlp_out`

Final: `gemma_rms_norm_add_dyn(mlp_out, residual)` → normed hidden → x.

## GEMV routing inside the emit

Mirror the serving crossovers (nvfp4/metal.py + scaled_mm/metal.py):
m == 1 → base kernel; even m ≤ 8 → `_mb`; odd 3 ≤ m ≤ 8 → `_mv4r`.
At the V1 eligibility bound (m ≤ 8) the quantized GEMV family covers
every projection; no dense-GEMM fallback inside the CB.

## Python wire-in

`Qwen3NextModel.forward`: on an eligible steady step with
`VLLM_QC_MUSE` on, skip the layer loop and call `muse_q38_run` (weights
registered lazily on first eligible step from the live modules — after
load-time repack, so pointers are the serving tensors). Shadow mode runs
both and logs per-layer boundary stats via `VLLM_QC_MUSE_LAYERS`
prefix capture.

## Phases

- **P1** — .mm scaffolding: init/layer/run + glue kernels
  (muse_expand_meta; reuse muse_sigmoid_mul/add/silu), build clean.
- **P2** — GDN-only prefix capture (layers 0..N all-GDN spans), shadow
  parity at m=4 serving shapes.
- **P3** — full-attn layers in-CB (rope/scatter/PA/gate), full 64-layer
  shadow parity.
- **P4** — serve-muse boot: needle, acceptance delta, legs, anchors
  re-gate; new pins; flip decision with measured TPS.

Risks: acceptance-rate roll (measured at P4, U47 precedent); PA
partition-count host decision inside one CB (encode-time choice from
seq_lens_cpu_max — steady steps make it stable); scratch aliasing bugs
(P2/P3 shadow catches); GDN state mutation ordering (conv_state/ssm_state
are written in-CB — muse must preserve the exact kernel-visible order the
eager path has: prepare reads+writes conv_state before recur writes
ssm_state; single encoder preserves program order within the CB).

## As-built deltas (what shipped vs the plan above)

The Why / Numerics stance / Phases sections above are accurate as written.
The two surface sections drifted during bring-up; the shipped ABI is:

- There is no `muse_q38_layer_common`: MLP weights and norm seams are
  registered through `muse_q38_layer_gdn` / `muse_q38_layer_attn` directly
  (both take `in_norm_w`, `post_norm_w`, `gu_t/gu_fmt`, `down_t/down_fmt`).
- `muse_q38_init` additionally takes `attn_scale`, `max_blocks`,
  `block_size`, `final_norm_w`, and `aux_layers` (the drafter's tap
  boundaries; the run emits `muse_add_out` aux exports at those layers).
- `muse_q38_run` additionally takes `residual_out`, `attn_group`,
  `attn_max_context`, `aux_out`, `debug_out`, `debug_layer`; the GDN
  metadata (`spec_cu`/`conv_slots`/`slot_table`/`num_accepted`) is a
  PER-LAYER tensor vector (the shared pools have per-layer slot windows),
  and the attention metadata (`block_table`/`seq_lens`/`attn_slots`/
  `attn_max_context`) is a PER-GROUP vector (the 16 attention layers span
  4 KV groups), with `attn_group` mapping layer -> group.
- `set_qproj` accepts a third format, fmt 2 = dense bf16 (`muse_dense_gemv`),
  for the projections Unsloth Dynamic leaves unquantized (in_proj_ba).
- The attention route picks split-K unconditionally at head_dim == 256
  (same predicate as the eager op), not by a sequence-length crossover.
- The sigmoid-gate glue is `muse_sigmoid_mul_exact` (per-op rounding
  mirror of the eager two-op chain), not a fused `muse_sigmoid_mul`; the
  SwiGLU emit is `qc_swiglu` (eager's own kernel) rather than a muse twin.
