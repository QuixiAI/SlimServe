# HANDOFF — LTX-2.5 on Apple Silicon: support + Metal kernel campaign

Branch `ltx25-metal` (worktree `SlimServe-ltx25`, from origin/main 6aa3c395).
Written 2026-10-02 at the close of the investigation phase and updated the
same day after bring-up (see Status). The other campaign handoffs (NVFP4-on-Metal,
MI300X GGUF, GLM-5.3-Flash on 8x A100) live in `HANDOFF.md`. Companion documents, read them in this order:

1. This file: mission, rules, decisions, milestones, gates, ops, risks.
2. `perf/ltx25_metal_campaign.md`: the evidence ledger (sections 1-12):
   physics, baseline table, precision gates, per-op profiles, ranked levers,
   h3.c survey, architecture decision. Every number cited here lives there.
3. `perf/ltx25_metal_research.md`: the web survey with URLs (projects,
   model facts, kernel prior art).
4. `perf/ltx25_harness/`: the measurement scripts that produced the ledger.
5. `perf/results/2026-10-01-ltx25-baseline/`: raw logs and frames.

## Status (2026-10-06)

Done and gated: N0 loader, N1 forward, N2 distilled end to end, the
output-preserving kernel-wave items that paid (split-K, fused glue, slab
VAE), N5 dev and DFR (default configuration), N6 profiles + CLI + serving.
Numbers: ledger sections 13, 15, 16 and 17 (16 = the remaining-headroom
accounting: output-preserving work on M1 Ultra is at its ~1% epsilon except
an attention kernel, ~1.5% short / ~5% HD). User and design documentation:
`docs/ltx25_metal.md`.

| clip | pipeline | baseline q8 | ours |
| --- | --- | ---: | ---: |
| 768x512x121 | distilled | 116.9 s | 86 s cold, 71 s resident |
| 768x512x121 | dev 30 steps | 606.9 s | 343 s |
| 768x512x121 | DFR | 170.9 s | 125 s |
| 1536x1024x121 | distilled | 504.9 s tiled | 343 s untiled |

Done since: the diffusion VAE decoder (default now, Metal neighborhood-attention
kernel, ledger section 18); section 20 audited the code against Lightricks'
source (`~/.local/scratch/ltx25/upstream`) and fixed the missing Gemma BOS
token, ancestral stage 2, STG block 28 and three dev deviations; section 21
added the keyframe-aware diffusion decode for DFR (joint NA Metal kernel,
parity 4e-7 vs upstream's torch reference), I2V first-frame conditioning (CLI
`--image`, API `image`), and re-validated dev and DFR at the default size.
Standing numbers at 1536x1024x121 after that and the section-22 decoder work
(QW=2 kernel, core-only queries, 48 GiB decode cache with the DiT parked):
distilled 406.6 s cold / 403.8 s resident, decode 97 s; DFR and dev measured
before section 22 at 807 s / 1540 s (their decodes shrink by the same ~70 s). Quality is only judged at
the default size (stage 1 must run at 768x512; section 20) and the parity
reference is upstream, never the dgrauet port (it drops BOS).
The fast tier (ledger section 24, 2026-10-06): three more profiles,
`ltx25-distilled-fast` / `ltx25-dev-fast` / `ltx25-dfr-fast`, the same
engine with output-changing settings (`engine.fast` -> `pipeline.Fast`:
schedule overrides, dropped guidance passes, the first-block step cache,
tiled stage-2 attention; plus the conv decoder). Measured one lever at a
time and stacked at 1536x1024x121 against the exact pipeline (PSNR + contact
sheets, clips in `~/.local/scratch/ltx25/n10/` for the user's review):
distilled 413 -> 278 s, dev 1513 -> 911 s, DFR 681 -> 421 s. Reviewed: no meaningful difference from exact; dev is clearly the best mode for quality and prompt adherence. The exact
profiles stay the reference; never tune them by the fast tier's method.
Done since (2026-10-06/07, ledger sections 25-28): the duration head (auto
clip length), DFR temporal rounds and the second spatial epilogue, the HQ
pipeline (res_2s, `ltx25-hq`, section 27), the prompt enhancer (Gemma-4
E2B-it in MLX, `enhance_prompt` / `--enhance-prompt`, section 28, with the
image-aware I2V variant through Gemma's vision tower, section 29). The
component checklist is complete. 2026-10-08: the user's standard is 100% of
upstream's capability, optimized the same way as the existing profiles;
section 30 is the inventory and order; sections 31-34 landed stills at any
frame on every profile, `ltx25-keyframes`, `ltx25-one-stage`, per-request
guidance / skip_step / user LoRAs, `ltx25-retake` (with the source media
path and both VAE encoders parity-gated) and `ltx25-ic-lora` (every control
IC-LoRA is a gated repo the account must accept before an end-to-end check),
each with a fast tier. Work the rest of section 30's list in its order.
Not done: the opt-in tier, any hand-written Metal GEMM
(measured headroom on M1-M4 is at most ~10% of a forward; see ledger section
15), N7 (candidates: na3d / na3d_joint kernels, split-K dispatch, slab conv
driver), HD runs of dev and DFR beyond the default size, nothing on parity: decoder (both modes, 120 dB), transformer forward and dev guided step (fp32 5e-6 / 2e-5) are all checked against upstream's own code run on this machine's CPU (`n8_*_parity.py`). Draft PR #86 on QuixiAI/SlimServe from fork
branch auroter:ltx25-metal (opened 2026-10-02; description carries the result
tables). Demo clip for the PR: not chosen yet; candidates in
`~/.local/scratch/ltx25/demo/v4/` (`beat1c_1536`, `beat1d_dfr_kf_1536`,
`i2v_beat1_1536`), user review pending.

How to run anything heavy: `perf/ltx25_harness/gpu_run.py --need-gb N -- <cmd>`
with `PYTHONPATH=<worktree>`. Environments: the engine and its tests run in
`~/.local/scratch/ltx25/venv-slimserve` (MLX 0.32.2 + the CLI's imports); the
registry suite runs in the main checkout's `.venv` (no MLX; the video tests
skip their MLX parts there); the baseline runner has its own venv. The main
SlimServe venv has no MLX: `requirements/video-metal.txt` lists what the
video profiles add.

## Mission

Add LTX-2.5 video+audio generation to SlimServe as a new subsystem, running
on Apple Silicon (M1 Ultra first, the whole M line as the target, M5 Ultra
buyers matter), and make it the fastest LTX-2.5 path on a Mac by as large a
margin as physics allows, without giving up quality. Three profiles, same
DiT, same kernels:

| id | pipeline (upstream class) | stage 1 | stage 2 | role |
| --- | --- | --- | --- | --- |
| `ltx25-distilled` | DistilledPipeline | 8 ancestral steps, half res, CFG 1 | 2x latent upscale, 3 steps, distilled LoRA | fast / iterate |
| `ltx25-dev` | TI2VidTwoStages | 30 steps x 4 guided passes (CFG 3.0, STG 1.0 on block 28, modality 3.0), half res | same | quality |
| `ltx25-dfr` | DFRPipeline | distilled + 5 generated keyframe slots | upscale, spatial-detailing epilogue with the official IC-LoRA (strength 0.5), optional temporal rounds (+8 steps each) | production (Lightricks' label) |
| `ltx25-hq` | TI2VidTwoStagesHQ | 15 res_2s steps x 3 guided passes, distilled LoRA 0.25 | 3 res_2s steps, LoRA 0.5, audio refined | Lightricks' HQ preset |
| `ltx25-keyframes` | KeyframeInterpolation | dev stage 1 with every still appended as a keyframe token block | dev stage 2, audio refined | a clip through given stills |
| `ltx25-one-stage` | TI2VidOneStage | 30 steps x 4 guided passes at the output size (snap 32) | none | prototyping |
| `ltx25-retake` | RetakePipeline | 8 distilled sigmas, plain Euler, at the source size; region mask on the tokens' time spans | none | edit a region of a clip |
| `ltx25-ic-lora` | ICLoraPipeline | distilled sigmas under the IC-LoRA with reference tokens, half res | bare 3 sigmas (adapter optional) | video-to-video |
| `ltx25-*-fast` | distilled, dev, dfr, keyframes, one-stage | output-changing settings stacked on (section 24 of the ledger) | | iterate, judged by eye |

Dev is the profile that adheres to the prompt (reviewed 2026-10-06: a lot
better than distilled and DFR); never pick a mode on speed grounds. DFR must be supported. Default output
1536x1024x121 @ 24 fps (stage 1 at 768x512); the campaign's short clip is
768x512x121 (stage 1 at 384x256).

Flow (standing): bring the engine up inside SlimServe, validate, then
upstream every kernel to QuixiCore-Metal (`<QuixiCore-Metal checkout>`,
`kernels/<family>/<op>/`, MLX + torch-MPS bindings) as a separate PR.

## Hard rules

- Commit identity: the contributor's repo-local
  `<name> <id+handle@users.noreply.github.com>` only (verify it in the
  worktree). No real emails, no attribution trailers, no AI credit anywhere
  public.
- Scope: the three ids above on platform `metal`; M5 variants are separate
  records later. Never widen a profile to a platform it was not validated on
  (`test_a_profile_is_one_config_per_platform`).
- Quality gates are output-preserving unless a flag says otherwise. Caching,
  tiling, sparse attention are an opt-in tier with their own A/B evidence.
- No large downloads, no multi-hour measurement queues, without asking.
  The user killed a 1.5 h baseline queue: measure what the decision needs.
- Durable state lives in the repo or `~/.local/scratch/ltx25`, never
  `/private/tmp`. Weights in `~/models/ltx-2.5`.
- Record every experiment before and after (perf/perf.md discipline): a
  change is a win only after correctness at stated tolerance, a measured
  improvement on real shapes, no regression on supported shapes.
- Run autonomously; no turn-ends at milestones; questions in plain text.

## Decisions (made with Sean, 2026-10-01/02; not open)

1. **Model variants**: official Lightricks bf16 checkpoints (dev and
   distilled DiT, 42 GB each), distilled LoRA rank 450 applied at runtime as
   a low-rank term (not fused: avoids a second 38 GB copy), Gemma-4 12B
   encoder bf16, conv VAE decoder (the diffusion decoder needs NATTEN and
   runs 10x slower eager on macOS; revisit after kernels), audio VAE +
   vocoder + BWE, spatial/temporal latent upscalers, duration head.
2. **Precision on M1-M4: no quantization.** fp16 MMA operands (every linear
   and attention), fp32 accumulate, fp32 glue (residual stream, norms, AdaLN
   tables and modulate, RoPE, softmax statistics, timestep/sigma embedding,
   x0 recovery) and an fp32 model boundary (latents, text embeds, timestep
   enter the DiT in fp32). Evidence (ledger section 9, identical captured
   inputs, fp32 truth): fp16 operands rel-L2 0.0071 at 13.7 s/forward; bf16
   operands 0.035 at 17.4 s; the baseline as shipped 0.042 at 32.1 s. The
   bf16->fp16 weight cast is bit-exact for these checkpoints. bf16 compute is
   emulated on M1/M2 (25-30% slower) and less accurate; on M3+ (native bf16,
   unverified on hardware) fp16 is still the choice for accuracy.
3. **Precision on M5**: official int8 convrot (W8A8, Hadamard g256, five
   bf16 islands) is the candidate for the int8 tensor units (Draw Things:
   1.6-1.9x int8 GEMM on M5). Custom calibrated quant only if the official
   one loses quality in an A/B. Official quants preferred: people trust them.
4. **Baseline**: dgrauet/ltx-2-mlx v0.15.12 (the engine under ltx-video-mac
   and Rapid-MLX; the antirez analog here). Bars: its q8 pack (recommended
   config) and bf16 pack. Secondary bar: ComfyUI on MPS (popular, 2-3.5x
   slower, broken on M1 Ultra today). Draw Things becomes a bar if it ships
   2.5 (it has real Metal kernels, stops at 2.3).
5. **Architecture**: SlimServe subsystem. Python orchestration on MLX;
   custom kernels in QuixiCore-Metal via its MLX binding. Rejected: torch-MPS
   host (slow linear path, residency/alloc-churn pain from the LLM campaign),
   C/Metal engine a la h3.c (months before first frame; h3 still runs
   MPSGraph for attention and conv).
6. **Reference trees**: the baseline clone (patched, `model.py.orig` is
   pristine) at `~/.local/scratch/ltx25/ltx-2-mlx` with venv; QuixiAI/h3.c at
   `~/.local/scratch/ltx25/ref/h3.c`; Lightricks/LTX-2 upstream (fetch files
   on demand; pipelines in `packages/ltx-pipelines/src/ltx_pipelines/`,
   constants in `utils/constants.py`).

## Standing numbers (M1 Ultra 64-core, 128 GiB, macOS 15.7.2, Xcode 26.3)

Physics: ~20.8 TF/s fp16 (fp16 is the only MMA on M1; fp32 GEMM is ~3.4x
slower), 800 GB/s. Measured MLX ceilings: GEMM 18.5-19.3 TF/s (K=4096) but
10 TF/s at K=16384; SDPA 15.1-15.3 TF/s fp16 (11.5 bf16). Weight traffic is
irrelevant (38 GB/forward = 48 ms); everything is compute-bound.

Baseline, seed 42, prompt "A red fox trotting through a snowy pine forest at
dawn..." (`perf/ltx25_harness/baseline_run.sh`):

| clip | pipeline | pack | wall | notes |
| --- | --- | --- | ---: | --- |
| 768x512x121 | distilled | q8 | 116.9 s | 4.3 s/fwd @1.5k tok, 17.8 s @6k, decode 18.1 s, 33.9 GB peak Metal |
| 768x512x121 | distilled | bf16 | 202.4 s | 8.4 s / 32.3 s per forward |
| 768x512x121 | distilled + our precision policy shimmed into the baseline | bf16 weights cast fp16 | **100.4 s** | 3.5 s / 13.4 s per forward; no custom kernels |
| 768x512x121 | dev two-stage 30 steps | q8 | 606.9 s | 17.2 s per 4-pass step |
| 768x512x121 | DFR | q8 | 170.9 s | 8 @ 2,016 tok (5.5 s) + 3 @ 9,600 (31.2 s) |
| 1536x1024x121 | distilled, `--tile-spatial 2` | q8 | 504.9 s | 20.7 s / 80.8 s per forward; decode 83.8 s at 53 GB; untiled decode kills the baseline |

Root cause of the baseline's slowness (ledger section 8): its fp32 AdaLN
tables promote the stream to fp32 and every DiT GEMM runs fp32. Our policy
is 2.36x on the forward and more accurate than the baseline's own numerics.

Projected engine at the measured ceilings, 768x512x121 distilled: ~73 s
(DiT 8x2.9 + 3x12, decode ~8 s with a 2x VAE, Gemma ~3 s, load ~3 s): 1.6x
over q8, 2.8x over bf16, before opt-in tiers. Honest ceiling: once the
precision path is in, the M1 Ultra DiT is within ~1.2x of its kernel
ceiling; the remaining large wins are the VAE, HD attention, fixed costs,
and the opt-in tiers.

Ranked levers (ledger section 10): (1) precision path, table stakes;
(2) conv3d VAE decode: 18% of the short clip, 84 s of 505 s at HD, memory
cliff; Draw Things' implicit GEMM is 2.4x on M1-M4, MLX's conv3d is
per-frame Winograd; (3) large-K GEMM (FFN proj_out at 10 TF/s, 15% of the
forward); (4) flash attention D128/D64 fp16 with fp32 stats + gated
epilogue (4% at 6k tokens, ~12% at 24k); (5) glue fusion (norm + modulate +
cast, gating, residual: 8-10%); (6) Gemma encode 8 s/prompt, load 3 s.

## Milestones and gates

**N0 Engine skeleton + loader (first).** `slimserve/video/` (or the name the
tree wants): official-safetensors loader for every component with a key map
documented against the runner's pack layout (mlx-forge is the converter to
read); config from `embedded_config.json` facts (48 layers, 4096/2048
hidden, 32 heads x 128/64, FFN x4 GELU no bias, qk RMSNorm, gated attention,
split RoPE max_pos [20,2048,2048] fp64 freqs -> fp32, AdaLN-single 9/2/5
param tables, patch 1x1x1, 128 latent channels, 32x/8x VAE). Gate: every
tensor loads, shapes match, dtype policy applied at load (fp16 weights,
fp32 tables), peak memory recorded.
**N1 DiT forward parity.** Our forward on the captured stage-2 inputs
(`perf/ltx25_harness/forward_gate3.py` writes `out/stage2_inputs.pkl`; the
runner's pytree) vs fp32 truth: rel-L2 <= 0.0071 (match the shim) and
<= 0.001 vs the runner's fp16-operand shim. Per-forward time <= 13.7 s at
6,144 tokens before any kernel. Deterministic across repeats.
**N2 Distilled pipeline end to end.** Text encode (Gemma-4 12B: mlx-lm has
no Gemma-4; the runner's `text_encoders/gemma/gemma4.py` is the reference;
feature extraction + projection + 8-layer connector), stage 1 ancestral
sampler with the exact distilled sigmas, upscaler, stage 2 with runtime LoRA,
conv VAE decode (tiled), audio VAE + vocoder + BWE, ffmpeg mux (brew ffmpeg
installed; conda `ldm` env also has one). Gate: same seed as the baseline
produces the same scene (latent cos vs runner-fp16-shim >= 0.99 at 49
frames; visual side-by-side), wall < 100.4 s on the short clip, peak memory
recorded, mp4 plays (121 frames @24, AAC).
**N3 Kernel wave 1 (output-preserving, bit-compared per op).** Conv3d VAE
decode (implicit GEMM, tiling planner, memory budget), large-K GEMM tile
config, fused rmsnorm+modulate+cast and gated-attention epilogue. Each
kernel: correctness vs MLX op at fp32 tolerance, standalone TF/s on real
shapes, forward-level delta, no regression on other shapes. Gate: short clip
<= ~80 s, HD clip untiled decodes without the memory cliff.
**N4 Kernel wave 2.** Flash attention (D128 self, D64 AV cross, text cross
with 1,024 keys, fp16 operands, fp32 statistics, gating fused), rope-in-
projection, residency/mmap no-copy, command-buffer overlap if MLX exposes
the need. Gate: HD stage-2 forward measurably below 67 s (1.2 PFLOP at 18
TF/s), short clip ~73 s.
**N5 Dev and DFR pipelines.** Guiders (CFG, STG perturbation on block 28,
modality), 4-way batched guided forward, LinearQuadratic schedule, dev
stage 2 with the distilled LoRA; DFR keyframe slots, segment-aligned canvas,
detailing IC-LoRA attach/detach, spatial epilogue tiling, temporal rounds,
Lanczos keyframe rebuild. Gate: parity with the runner on each (same seed
scene), dev short clip ~375 s projected, DFR measured; then the dev-vs-DFR
quality A/B (fixed prompts, seeds, side-by-side) that decides which id the
docs call "best quality".
**N6 Profiles + CLI + serving.** Three records in `profiles.json` under a
new source type, `slimserve ltx25-dev --dry-run/--serve`, request queue
(one GPU: serialized requests with resident weights; concurrency is part of
done: N concurrent requests complete correctly, memory stays flat, no
watchdog kills), fetch of gated weights (HF token must have accepted
Lightricks/LTX-2.5 and the IC-LoRA repo), smoke tests, docs.
**N7 QuixiCore-Metal PR.** Every kernel under `kernels/<family>/<op>/` with
MLX + torch-MPS bindings, `kernels.yaml` entries, correctness tests, bench
entries. Separate PR, after the SlimServe PR.
**Opt-in tier (after N6):** TeaCache/FBCache (baseline has TeaCache,
1.46-1.78x), spatial tiling (baseline: 111 -> 81 s per HD forward), sparse
attention at HD; each with its own quality A/B. M5: int8 convrot path,
native bf16 check, NAX GEMM; needs an M5 box.

## Component checklist (nothing skipped)

DiT: patchify proj (128->4096, audio 128->2048), timestep sinusoidal + MLP,
AdaLN-single tables (9 video self-attn, 2 prompt, 5 AV-CA + gate), per-token
timestep variant, 48 x BasicAVTransformerBlock (self-attn + qk-norm + split
RoPE + gated attention; text cross-attn with prompt AdaLN; bidirectional AV
cross-attn with cross-modality AdaLN and gates, temporal-only RoPE; GELU
FFN), output AdaLN + proj_out, x0 recovery in fp32, STG perturbation hook,
attention masks (keyframe slots, prompt relay), keyframe absolute embedding.
Text: Gemma-4 12B fine-tune, multi-layer feature extraction, projection,
connector transformer (8 layers, 32x128, 128 registers, gated attn, RoPE
max_pos 4096) -> 4096-d video context, 2048-d audio context; negative
prompt path for dev. Duration head. Prompt enhancer (Gemma-4 E2B-it, text; opt-in).
VAE: causal conv3d encoder (image/video conditioning), conv decoder
(timestep-conditioned), tiled decode with budget, latent normalize /
denormalize, spatial x2 and temporal x2 latent upscalers. Audio: mel
processor, 2-D causal conv VAE, HiFi-GAN vocoder (snakebeta, upsample
[6,5,2,2,2]), bandwidth extension. LoRA: runtime low-rank application
(distilled rank 450, detailing IC-LoRA), strength. Samplers: Euler
ancestral (distilled stage 1, seeded noise offsets), Euler, res_2s (HQ),
LinearQuadratic schedule, sigma lists from upstream constants. Guiders:
CFG, STG, modality, rescale 0.7. I2V conditioning (image CRF 33 re-encode).
Mux: ffmpeg H.264 + AAC. Memory: residency, mmap no-copy weights, arena
reuse, VAE budget.

## Ops constraints and gotchas (each cost time already)

- HF token on this machine is `<hf-account>`; the gated acceptances live on
  that account.
  Downloads: use `HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1`
  (xet path was 1 MB/s per file; hf_transfer 20+ MB/s).
- Detached long jobs: `nohup` fails under this shell; use
  `subprocess.Popen(..., start_new_session=True)` from python.
- Never hold two dtype copies of the DiT in one process (the fp32 copy next
  to the original put the box 5 GB into swap and corrupted timings). One
  config per process; reload from disk (2 s mmap).
- Baseline GPU runs must be serial; a concurrent GPU job halves measured
  rates. The download process is CPU-only and safe.
- **One model-loading process at a time, always through
  `perf/ltx25_harness/gpu_run.py --need-gb N -- <cmd>`.** On 2026-10-02 at
  11:07 the box kernel-panicked (`watchdog timeout: no checkins from
  watchdogd in 90 seconds`) and had to be forced off. Cause: three parallel
  component-port agents plus the lead each ran model-loading jobs; the
  JetsamEvent reports show python processes at 84.7, 64.2 and 18.1 GiB
  resident (167 GiB on a 128 GiB box) with a fourth loading the 39 GiB DiT,
  `iogpu.wired_limit_mb` at 122880. Metal memory is wired: it cannot be
  compressed or swapped, so the compressor ran out of space
  (`vm-compressor-space-shortage` killed tccd, trustd, logd_helper, ...) and
  watchdogd starved. Rules: never run parallel agents that touch the GPU or
  load weights (parallel agents may read and write code only); every heavy
  run goes through the guard (exclusive lock, headroom check, 16 GiB OS
  reserve); parity scripts load one model copy per process and compare via
  saved .npz files, never reference + candidate in one process.
- `iogpu.wired_limit_mb` resets to 0 (the macOS default, about 3/4 of RAM)
  at boot. Leave it there unless one resident configuration measurably needs
  more, and never above 110000 on this box; the guard refuses to launch
  above that. The old 122880 setting left the OS under 8 GiB.
- Untiled VAE decode at 1536x1024x121 estimates 56 GB and killed the
  baseline process silently (only a leaked-semaphore warning).
- The runner's CLI needs ffmpeg on PATH; `brew install ffmpeg` done
  2026-10-01.
- Comparing pipeline outputs across runs measures sampler divergence, not
  kernel error (8 ancestral steps turn 1e-3 into cos 0.95). Per-forward
  comparisons on identical captured inputs are the precision instrument.

## Risks and unknowns

- Gemma-4 architecture port (no mlx-lm support): sized by the runner's
  implementation; budget it in N2.
- Official safetensors key layout vs the runner's converted pack: loader
  mapping work in N0; mlx-forge documents the conversion.
- DFR details (keyframe slot canvas, epilogue tiling constants) come from
  upstream `dfr_helpers/`; the runner calls its DFR experimental.
- Native bf16 on M3+ and all M5 claims are unverified on hardware here.
- Draw Things' conv3d kernel source is only partially public; ours is
  original work with the h3.c tiling planner as a pattern.
- Quality A/B needs a judge: fixed prompt set, same seeds, side-by-side
  frames, plus latent agreement vs the runner; no automatic metric is
  trusted alone.
