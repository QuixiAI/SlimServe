# Drafter Requirements for 100 tok/s (Muse-Glimmer-30B, M5 Max)

2026-08-15. Companion to `perf/optimization_status.md` entries (9)-(11) and
the revised ledger in `HANDOFF.md`. This memo turns the measured infeasibility
of 100 tok/s with the current DFlash drafter into a concrete requirement for
a replacement artifact, so the training-side decision can be made with
numbers.

## Why the current drafter caps the campaign

- Serving physics: one weight pass is ~38.6-43 ms (18.9 GB at the measured
  ~460-500 GB/s). With every serving-side leg landed perfectly, the verify
  step bottoms out at ~43-48 ms. 100 tok/s therefore requires
  **E[tokens/step] >= 4.3-4.8**.
- Measured drafter profile (conditional acceptance by draft position,
  greedy, 3-prompt mix): **0.775 / 0.647 / 0.374 / ~0.45 / ...** ->
  E = 2.59 (predicted) vs 2.46-2.7 (observed; model validated).
- Trees cannot rescue a weak drafter: offline upper-bound replay of the
  drafter's top-8 across 234 real steps caps buildable (<= 48-node) trees
  below ~3.0 tokens/step and even a 611k-node tree at 3.47. The failure is
  drafter signal beyond position ~3, not verification topology.

## The requirement

| Serving step | E[tokens/step] needed | Constant conditional acceptance |
| ---: | ---: | ---: |
| 43 ms (physics floor) | 4.3 | ~0.775 sustained |
| 48 ms (near-floor, realistic) | 4.8 | ~0.80 sustained |
| 55 ms (conservative) | 5.5 | ~0.83 sustained |

The current drafter already hits 0.775 at position 1. **The requirement is
not a better first guess — it is holding ~0.78-0.80 conditional acceptance
through 8-10+ positions instead of collapsing after 2.** A tree verifier
(cheap on the M5 tensor units: M=48 verify ~ +76% MMA, MMA off the critical
path) relaxes the linear requirement by ~10-15%, i.e. E ~ 4.3 linear plus a
small tree reaches the 48 ms target.

## Cost envelope for a bigger drafter

The drafter forward streams its weights every step. Budget if the step is
to stay near-floor:

- Current: 1.62 GB (5 layers), forward measured 12.6 ms (3.6x its 3.5 ms
  stream floor; fusion work pending would bring it toward ~5 ms).
- A drafter up to ~3-4 GB remains viable IF its forward is fused/efficient
  (~8-10 ms) AND it buys E >= 4.3: e.g. at E = 4.8 a 10 ms drafter inside a
  55 ms step still yields ~87 tok/s; at E = 5.5 -> ~100.
- Deeper conditioning (more aux layers, larger block training, DSpark-style
  confidence truncation) matters more than parameter count per se: the
  collapse is positional, which suggests the block-denoising training did
  not force deep-position fidelity.

## Acceptance test for ANY candidate drafter (run BEFORE integrating)

`perf/results/2026-08-15/plain-step-decomp/tree_acceptance_study.py` is the
harness: it logs the candidate's per-position top-8 against true greedy
continuations on a 3-prompt mix and reports tokens/step for linear and tree
topologies (the k=1 row must reproduce the drafter's production acceptance;
alignment advances by real linear emission). Gate: **linear E >= 4.0 and
48-node-tree E >= 4.5 on this harness before any serving integration.**
The serving stack (tensor-ops verify kernels, native input prep, spec
plumbing) transfers unchanged to any DFlash-compatible artifact.

## Addendum (2026-08-15): precision eliminated as a factor

The published bf16 original of the same 5-layer drafter
(meta-models/Muse-Glimmer-30B-assistant) measures identically on the
acceptance harness (linear 2.52 vs 2.46 quantized; trees within noise).
The positional collapse is architectural, not quantization. A retrained
artifact is required; quantizing it to Q4_K costs ~nothing in acceptance.

To test a candidate safetensors drafter with the harness: point
speculative_config.model at its directory (DRAFTER=bf16 branch in
tree_acceptance_study.py shows the pattern) and, if it ships published
naming, the fork now handles it; config.json may need: model_type
"qwen3", architectures ["DFlashMuseGlimmerDraftModel"], explicit
vocab_size 202048, dflash_config.swa_window_size 2048.
