# Research: where the precedent lives

Research is the first step of every item and the first day of every
campaign. Its output is a digest in HANDOFF.md: **copy this** (with the
source and what transfers to this hardware) and **avoid that** (measured
negatives, with where they were measured). Time box: one hour per item, one
day for the campaign's opening pass.

## Online (verify negatives here first)

Ecosystem claims go stale in days. Before resolving any plan on "no drafter
exists", "unsupported upstream", "nobody has published speeds", search:

- The quant author's engine and repo (for GGUF quants by antirez: `ds4`,
  its `PERFORMANCE.md`, blog posts, open PRs; he writes custom Metal / CUDA
  kernels for his own quants - read the kernel that serves the exact tensor
  layout). Clone upstream into `~/.local/scratch/<engine>-upstream`, never
  use a stale local snapshot.
- Model support and optimization PRs, by model name and architecture:
  `gh search prs --repo vllm-project/vllm "<model>" --limit 50`,
  `gh search prs --repo sgl-project/sglang "<model>"`,
  `gh search prs --repo ggml-org/llama.cpp "<arch or quant>"`,
  plus ik_llama.cpp, mlx-lm, FlashInfer, DeepGEMM, TensorRT-LLM, kernel
  libraries for the platform (AITER for ROCm, Marlin / CUTLASS / Machete
  for CUDA, MLX and ggml Metal for Apple). Read the kernel diffs, not the
  PR titles: layouts, fusion boundaries, tile geometry, dequant tricks,
  launch counts, acceptance numbers.
- Published speeds for this model and quant on this hardware class
  (blogs, README tables, speed-bench CSVs, HF model card discussions).
  Record the protocol beside every number; most are not our shape.
- Drafters for the model: MTP heads in the checkpoint, DFlash / DFlash2,
  DSpark, EAGLE-3, published on HF (`incoai`, `z-lab`, the model lab's
  org). Licenses noted; the fastest one on this box ships.
- Upstream vLLM issues for the model (acceptance bugs, parser defaults,
  hybrid-model prefix-caching defaults) so known defects are not
  rediscovered.

Use WebSearch and WebFetch; cite the URL and the commit or date in the
digest.

## In the repo (the campaigns before this one)

- `grep -n '^## ' perf/optimization_status.md` - every experiment on every
  platform, with its verdict. Read the entries for the same model family,
  the same kernel class (MoE GEMV, sparse MLA, GDN / KDA recurrences, mHC,
  routers, samplers) and the same platform, in that order.
- `perf/*retrospective*.md`, `perf/*campaign*.md`, `perf/*design*.md`,
  `perf/*handoff*.md`, `docs/*.md`: ceiling analyses, instrument playbooks,
  cross-platform lessons ranked by step time recovered.
- HANDOFF.md campaign sections for the other platforms of this model:
  what exists on `main`, the blocker lists with file:line, the traps.
- `slimserve/profiles.json`: the closest existing record (same model on
  another platform, or the same platform on the previous model) is the
  template for engine args, env opt-ins, speculator and notes.

Techniques transfer across hardware more often than they look like they
should: launch fusion, weight-stationary small-M GEMVs, in-place shard
writes, expert-grouped decode routes, batch-adaptive speculation schedules,
residency pinning, single-command-buffer or graph-captured steps. For each
candidate write one line: technique, where it was measured, what it
recovered, what changes on this box.

## Reference trees on the machine

Check which of these exist (`ls -d`) and read them for layouts, bounds and
kernel behavior: `~/llama.cpp`, `~/ds4` (may be a stale snapshot; prefer
`~/.local/scratch/ds4-upstream`), `~/QuixiCore/*` or `~/Code/QuixiCore-*`
(the kernel library SlimServe vendors from and ports back to), the vLLM
Marlin and FlashInfer sources in the venv.

## What the digest must answer before bring-up starts

1. What is the bar engine, at what version, with what best config, and what
   does it do per token (bytes, launches, kernels) that we must match or
   beat?
2. Which of our existing kernels apply unchanged, which need a parameter
   change (top-k, head dim, expert count), which are missing?
3. What is the per-token byte budget and the bandwidth ladder on this box
   (measured single-kernel ceiling, sustained ceiling, current)?
4. Which drafters exist, and what acceptance do others report?
5. Which techniques from the other platforms' notebooks transfer, ranked by
   expected step time recovered?
