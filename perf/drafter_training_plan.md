# Drafter Retraining Plan — the Gate to 100 tok/s

2026-08-15. Companion to `perf/drafter_requirements.md` (the WHAT) — this is
the HOW, costed against this machine and the measured campaign data. Written
so the go/no-go decision is concrete.

## Objective

A DFlash-compatible drafter for Muse-Glimmer-30B with conditional acceptance
~0.78–0.80 sustained through positions 1–10 (current artifact: 0.775 / 0.647
/ 0.374 / collapse). Gate before any serving integration: **linear E ≥ 4.0
and 48-node-tree E ≥ 4.5 on the replay harness**
(`tree_acceptance_study.py` — runs in minutes per candidate).

## Why the current drafter collapses (evidence-based diagnosis)

- Precision is ruled out (bf16 == Q4_K on the harness).
- The collapse is positional: the 5-layer block-denoiser predicts position
  1 nearly as well as the target agrees with itself, then decays fast —
  the signature of insufficient DEPTH-WISE conditioning in block
  denoising: deep positions see only mask embeddings + aux features, and
  5 layers of SWA cannot propagate enough sequential structure.
- Levers, in expected-value order: (1) more layers (8–10 vs 5) — depth is
  what deep positions lack; (2) per-position loss weighting emphasizing
  positions 3–10 (the published recipe optimizes mean CE, which the easy
  early positions dominate); (3) multi-round denoising at train AND serve
  time (2 passes: draft, re-embed, re-draft — serving cost +1 drafter
  forward ≈ +12 ms, acceptable if E gains ≥ 0.5).

## Architecture

Extend the published `MuseGlimmerAssistantModel` shape (hidden 6656, GQA
32/8, SWA 2048, block 16, aux layers [1,13,25,37,49]) from 5 → 9 layers:
~4.4B params, bf16 ≈ 8.8 GB. Serving cost when quantized to Q4_K ≈ 2.6 GB
streamed/step ≈ +2.3 ms over today's drafter — inside the budget from the
requirements memo (E = 4.8 at a 55–60 ms step still clears 85–95 tok/s;
with the serving roadmap's remaining wins, 100+).

Initialize layers 1–5 from the published bf16 assistant (now loadable in
this fork), duplicate-and-perturb for the new layers (depth-upscaling), keep
the shared embedding/lm_head contract.

## Data

Teacher-forced distillation does NOT need target generation throughput —
it needs target FORWARD passes over existing text (prefill ≈ 1200 tok/s
measured). Recipe per sample: run the target over a document chunk,
capture aux hidden states + next-token distributions; train the drafter to
predict blocks of 16 continuations at every position (standard DFlash
distillation). Data: ~2–5B tokens of natural text + agentic traces
(the model's domain). At 1200 tok/s teacher prefill on this machine:
2B tokens ≈ 19 days — TOO SLOW LOCALLY for the teacher pass at full scale.
Options:
  (a) Rent one 8×A100/H100 node for the teacher-capture + training
      (~3–7 days wall, the standard choice; the fork's CUDA path serves
      the target for capture).
  (b) Local-only reduced scale: ~300–500M tokens ≈ 3–5 days of capture +
      ~2–4 days of MPS training (torch MPS trains a 4.4B bf16 student on
      128 GB; optimizer states ~35 GB, activations manageable at short
      blocks). Risk: may undershoot the acceptance gate; mitigated by the
      cheap offline gate — measure E after each 100M tokens and stop
      early on plateau.

- Checkpoint cadence: run the replay harness per checkpoint (minutes);
  training continues only while E climbs.

## Timeline and decision

- Local path (b): ~1–1.5 weeks machine-dedicated, zero cash, moderate risk.
- Rented path (a): ~1 week including setup, standard risk, cluster cost.
- Either path ends with: quantize (Q4_K costs nothing per the precision
  measurement), run the acceptance gate, then the serving stack from this
  campaign applies unchanged — at E ≥ 4.8 the existing measured roadmap
  arithmetic reaches 100 tok/s at short context.

## What is NOT being asked

No training has been started. This plan exists so the decision is
"approve (a) or (b) or neither," with the gates, costs, and the measured
reasons the current artifact cannot be salvaged all in one place.

## Addendum: feasibility spike MEASURED (2026-08-15)

The pipeline exists and runs (`perf/drafter_training/`):

- `capture_teacher.py`: teacher aux-hidden capture via layer hooks over the
  serving stack. Measured **232 tok/s** capture (hook + per-chunk engine
  overhead; batching should reach ~400-600). Aux payload is 66.6 KB/token
  -> a real corpus CANNOT be stored (20 TB at 300M tokens): the pipeline
  must stream teacher -> student in-process (capture a chunk, train on it,
  discard).
- `train_spike.py`: compact trainable reference of the drafter
  (plain-PyTorch, checkpoint-compatible -- all 58 published bf16 tensors
  load with zero unexpected keys; depth-extension leaves exactly the new
  layers to initialize). Forward+backward on MPS: **545 tok/s at 5
  layers, 468 tok/s at the target 9 layers**, loss decreasing.
- Measured local single-machine pipeline (serial capture+train on one
  GPU): ~155 tok/s combined -> 100M tokens ~= 7.5 days, 300M ~= 22 days
  (optimizable toward ~13-15). The RENTED path is the fast lane if
  calendar time matters; the LOCAL path is real but machine-dedicated for
  1-3 weeks.
- Still unbuilt (deliberately -- awaiting go/no-go): the streaming
  glue between the two scripts, the true DFlash block-denoise loss
  (teacher top-K CE per block position with position weighting), and the
  checkpoint-gate automation. Each is bounded engineering on top of the
  proven pieces.

## Addendum 2: streaming trainer COMPLETE and smoke-validated (2026-08-15)

`perf/drafter_training/stream_train.py` is the full pipeline: serving
target captures aux hiddens + final hiddens per chunk, teacher top-K
computed via the shared lm_head, student trained on the true block-denoise
objective (per-position-weighted CE against teacher top-K; weights 1/1/2..2
/3..3 targeting the measured collapse zone). Smoke (40 steps, repo-doc
corpus, 9-layer student): loss 11.05 -> 9.70, monotone by decade.
**Measured combined pipeline: 124 tok/s** -> 100M tokens ~= 9.3
machine-dedicated days local; 300M ~= 28 (before capture batching
optimizations). A real run is: a corpus path + STEPS=-1 + checkpointing
(the one remaining glue line item) -- the go/no-go decision now approves
an actual command, with the E-gate harness ready to score checkpoints.
