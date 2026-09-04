# GLM-5.3-Flash on 4x RTX PRO 6000 Blackwell (sm_120): performance plan

Status: plan, 2026-09-04. Branch `glm53-flash-sm120`, cut from upstream/main
9247eedad (2026-09-04). No code changes yet. Companion notebook entries go in
`perf/optimization_status.md`; baselines in `perf/baseline_status.md`; raw
logs under `perf/results/YYYY-MM-DD/<run-id>/`.

Goal: an optimized SlimServe profile `glm53-nvfp4-4` for the platform
`rtx6000` (4x RTX PRO 6000 Blackwell Workstation, sm_120, PCIe 5 x16, no
NVLink, 96 GB per card) that runs GLM-5.3-Flash as close to the bandwidth
floor as the engineering allows, measured every step, with no regressions.

## 1. Hardware and model facts that set the ceiling

RTX PRO 6000 Blackwell Workstation (measured on tinybox, driver 580.173.02):

| item | value |
|---|---|
| SMs / L2 / smem per SM | 188 / 128 MiB / 100 KB (99 KB opt-in per block) |
| memory | 96 GB GDDR7, 512-bit, 14 GHz -> 1.79 TB/s |
| max clocks / power | 3090 MHz SM, 600 W (stock) |
| topology | GPU0-GPU1 and GPU2-GPU3 PHB, cross-pair NODE; P2P read OK on all pairs |
| CUDA | 12.8, 13.0, 13.2 toolkits; family cubin `12.0f` needs CUDA >= 13.0 |

GLM-5.3-Flash (`/raid/weights/GLM-5.3-Flash/config.json`, native FP8, 306 GiB):
45 layers, first 3 dense (FFN 12288), 42 MoE layers with 288 routed experts
(top-8, `moe_intermediate_size` 2048 -> 25.2M params per expert) plus one
shared expert; 34 KDA linear-attention layers (64 heads x 128, state 2 MiB
per layer per sequence) and 11 DSA sparse-MLA layers (64 heads, 256 nope
dims, no RoPE, `kv_lora_rank` 512, indexer 32 heads x 128 with 4x key
pooling, top-2048); mHC hyper-connections; vocab 154,880; one MTP layer;
1M positions.

Bytes per token at batch 1 (routed experts 8.46B params; non-expert
"backbone" ~9.2B: KDA 5.1B, DSA attention 2B, shared expert 1.06B, dense FFN
0.45B, lm_head 0.63B):

| weights (measured from the checkpoints, 2026-09-04) | experts/GPU | backbone/GPU | per GPU at TP4 | floor per step | no-spec c1 ceiling |
|---|---|---|---|---|---|
| NVFP4 experts + BF16 backbone (RedHatAI as shipped) | 1.19 GB | 4.59 GB | 5.78 GB | 3.2 ms | ~310 tok/s |
| + native FP8 for what ZAI quantized (dense MLP, shared experts, DSA q_a/q_b/kv_a/o) | 1.19 GB | 3.87 GB | 5.05 GB | 2.8 ms | ~355 tok/s |
| + self-quantized FP8 KDA q/k/v/o (ZAI kept BF16; quality gate required) | 1.19 GB | 2.72 GB | 3.91 GB | 2.2 ms | ~460 tok/s |
| + kv_b and lm_head FP8 | 1.19 GB | 2.52 GB | 3.71 GB | 2.1 ms | ~480 tok/s |
| NVFP4 everything | 1.19 GB | ~1.3 GB | ~2.5 GB | 1.4 ms | ~720 tok/s |

Per-GPU figures follow SlimServe's sharding (column/row-parallel /4;
norms, mHC, router gate, DSA q_a/kv_a, indexer, KDA f_a/g_a replicated).
At TP4 the backbone, not the experts, dominates the per-token read: KDA
q/k/v/o alone (2.28 GB/GPU, BF16 in the native checkpoint too) is about 2x
the routed-expert read. Section 3d has the tensor-level inventory.

Communication floor at TP4: about 90 all-reduces per step (2 per layer).
Measured 2026-09-04: pynccl ring LL is ~11 us each at c1 (0.93 ms/step)
and ~15 us at c8 (1.39 ms/step); a one-shot PCIe-IPC reduction might halve
that, no more.

MoE speculation math: every verified draft token and every concurrent
request routes to up to 8 more experts per layer, so draft length and batch
compete for the same expert-bandwidth budget. At ~40 tokens per step the
step touches nearly all 288 experts and step time stops growing (~25 ms per
step at NVFP4, 44 GB per GPU). With FP8 backbone and MTP-3 (about 3 accepted
tokens per step, ~30 distinct experts per layer): c1 floor ~4.8 ms including
comm -> ~630 tok/s; c8 floor ~15.4 ms for 24 tokens -> ~1550 tok/s.

Engineering target = 60-70% of floor:

| shape | today (Foundry SGLang FP8) | B12X R24 bar (4x PRO 6000) | B12X R24 on tinybox, handicapped | our Phase 0 baseline | target |
|---|---|---|---|---|---|
| c1, no spec | 88.5 | 169.9 | 134.0 | 104.8 | 300-350 |
| c1, MTP-3 | 141.7 (NextN 5/6) | 247.8 (2.50 acc/step) | 194-212 (2.2-2.8 acc/step) | - | 400-500 |
| c8, no spec | - | 737.8 | 483.7 | 431.4 | 900-1000 |
| c8, MTP-3 | 274 at c4 | 903.2 | 477-509 | - | 1000-1200 |
| c16, no spec | - | - | 598.6 | 591.0 | - |
| prefill at 32K | not measured | 14.9K tok/s | not measured | not measured | >= 15K |

The handicapped column is the published image run on this host with its TP
all-reduce forced onto a PyNCCL ring over host shared memory (the 580.173
driver is CUDA 13.0; the CUDA 13.3 image's NCCL cuMem/P2P and B12X CUDA IPC
paths fail, launcher knob B12X_PCIE_ALLREDUCE=0; notebook entry "B12X R24
control on tinybox (handicapped)"). It is a floor we must beat in every cell,
not the bar; an unhandicapped local control needs a 13.3-capable driver,
which is an operator decision on the shared box.

TP2 halves sync count but doubles bytes per GPU and leaves no KV room at
131K context; TP4 with TP-sharded experts is the layout (EP measured 4-6%
slower on A100, and B12X also serves TP4/DCP1).

## 2. Where the tree stands (upstream/main 2026-09-04)

Present, A100-validated: `vllm/model_executor/models/glm5_next.py` (model,
mHC sites, vision, torch.compile), Triton KDA (`kimi_gdn_linear_attn.py`,
`third_party/flash_linear_attention/ops/kda.py`), pooled DSA indexer
(`glm5_next_indexer.py`), QuixiCore sparse NoPE MLA decode
(`csrc/quixicore/serving/mla_kernels.cuh`, `py_mla_decode_bf16_sparse_nope`,
partitioned P128 with the 512 MB scratch gate), mHC ops on the DSV4 Ampere
kernels, NVFP4 experts on Marlin W4A16, host + NVMe KV tier. Profiles
`glm53-nvfp4-4` / `-8` (a100 only). A100 TP4 record: 73.8 / 332.1 / 464.6
tok/s at c1/c8/c16 (1000 in / 300 out, temp 1.0, seed 42).

Missing for this work: no `rtx6000` platform (`hardware.py:_classify`,
`profiles.json` platforms and per-profile variant), no sm_120 QuixiCore
kernel (all `*_ampere`), no native NVFP4 tensor-core MoE on the serving path
(Marlin W4A16 only), MTP draft path not ported (`profiles.json` speculator
note), no PCIe all-reduce, backbone served in BF16.

A100 TP8 profile ranked residue (transfers directly, it is launch count):
~150 `direct_copy` launches per token (0.5 ms), pooled indexer 65 us -> 15
us (0.5 ms), mHC channel ownership (up to 1.2 ms), the 280-launch M=1 GEMV
chain (KDA's four projections).

## 3. Research digest, 2026-08-30 to 2026-09-04

### Borrow

- vLLM #53906 (merged 09-03): native GLM-5.3-Flash incl. the MTP draft
  model. Source for the SlimServe MTP port. #55214 adds the missing
  `__init__.py`.
- vLLM #55277 (open, tested 4x RTX PRO 6000 TP4) + flashinfer #4802
  (merged 09-03, `GLM53_NOPE`, decode 14.9 -> 11.3 us) + flashinfer #4947:
  the SM120 sparse NoPE MLA path and the `fp8_ds_mla` row layout for
  pe_dim = 0. Candidate replacement or A/B partner for the QuixiCore kernel.
- vLLM #53576 (merged 09-01, validated 4x RTX 6000 Pro TP4):
  FlashInfer PCIe-IPC all-reduce, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`,
  needs FlashInfer > 0.6.17. flashinfer #4870 (open): copy-engine variant.
  B12X's own one-shot: below 56 KB payload, +7% decode at TP4, same-NUMA only.
- KDA: flashinfer #4709 (merged 09-03) output-only KDA decode 1.5-5.9x;
  vLLM #55364 (open) FlashInfer KDA prefill/decode, 2.47x decode at bs1;
  vLLM #54697 (merged) overlaps low-M KDA projections; SGLang #37744 (open)
  qkvbfg fusion 6 -> 2 GEMMs per KDA layer on the FP8 checkpoint.
- local-inference-lab/vllm #576 (closed, unmerged): `cp.async.bulk.prefetch.L2`
  of next weights during reduction windows, in_proj 29.9 -> 17.1 us, +7% c1.
  Same idea as our L2-prefetch item; 128 MiB L2 holds a layer's backbone.
- vLLM #55170 (open): CUTLASS W4A4 over W4A16 for NVFP4 linear layers on
  SM120, +23.8% tok/s on RTX PRO 6000 Max-Q. flashinfer #4718 (merged)
  sm12x NVFP4 MoE retune; #4013 (merged) drops tiles over 101 KB smem.
- vLLM #54110 (open, approved): persistent top-k fallback for < 128 KiB
  smem GPUs, tested with GLM-5.3-Flash MTP. #54951 (draft): indexer prefill
  sharded across TP, -23% TTFT at 1M.
- llama.cpp #27970 (merged): gather-then-dense FA over indexer-selected KV,
  2.2x at 1M. llama.cpp #27917 (draft MTP): per-position acceptance
  0.92 / 0.81 / 0.60 / 0.44 / 0.35 -> depth 3 is the sweet spot.
  abtraore/GLM53-BLACKWELL: DSA-pool graph caching, NextN head, SSM placement.
- exllamav3 #330: BF16 I/O on the decode GEMM, +11.6% at c4 from removing
  dtype conversions. siddharth1825 trellis MoE prefill: 288 experts per launch.
- B12X R24 recipe: two MoE paths by concurrency (fused lowest-latency for
  c1-2, CUTLASS for c4-8), FP8 KV, full CUDA graphs, PREFILL_COMPUTE_SHARE.
- SGLang issue #37813 (ormandj): the SM120 blocker list and a working 2x TP2
  image (154 tok/s c1, 450K ctx, FP8 KV) with unfiled patches: PCIe-IPC AR in
  GroupCoordinator, KDA gate fusion, FP8 lm_head.
- Research: DASC (arXiv 2608.30386) KDA state compression.

### Avoid

- The localmaxxing 1005 / 1206 / 1494 tok/s runs are "locked attractor"
  n-gram measurements with 8K KV. Not a target.
- Re-writing cuBLAS-class BF16 skinny GEMMs on sm_120: SGLang #37899
  measured +4% op-level on RTX PRO 6000 and ~0% end to end. The lever is
  launch count and fusion, not per-kernel bandwidth on dense BF16.
- ModelOpt NVFP4 checkpoints (LibertAIDAI, dealignai) emitted invalid
  UTF-8 on 4x RTX PRO 6000 in stock vLLM (issue #54150). RedHatAI
  compressed-tensors was clean. Any checkpoint gets the text gates first.
- Torch's P2P assumption is wrong for some PCIe sm_120 copies (exllamav3
  #333 gibberish on 2x PRO 6000). Validate every peer copy path explicitly.
- KDA inverse numerics: FlashKDA R21 emitted 55K non-finite values under
  concurrency; R22 uses FP32 forward substitution. Keep FP32 there.
- Reduction order changes MTP acceptance (SGLang #37746: 3.79 -> 5.88
  after keeping custom AR; #37745 closed). Measure acceptance before and
  after any all-reduce change.
- Speculation + hybrid KDA/MLA cache under mixed workloads made TTFT
  alternate 2x on 2x GB10. Qualify spec under the mixed lifecycle leg.
- vLLM #54458: hybrid page alignment inflates attention blocks to 7808
  tokens and collapses KV capacity and prefix-cache hits. Check block
  geometry on the port.
- smem: TileLang v2 DSA schedules exceed ~100 KB; FlashInfer dropped
  tiles over 101 KB; persistent top-k needs the < 128 KiB fallback.
- SGLang cookbook Blackwell presets (trtllm DSA, deep_gemm) do not run on
  sm_120. Not a reference config.
- flashinfer #4827 (open): sm12x MoE workspace use-after-free under CUDA
  graphs, repro GLM-5.3-Flash TP2 on RTX PRO 6000. Gate any FlashInfer MoE
  path on a graph-replay soak.
- llama.cpp glm5next PRs (three competing, unmerged, ~7% loss from no
  graph reuse) and ik_llama.cpp: not serving paths for this box.
- incoai DFlash2 drafter is CC BY-NC-ND and lost to MTP-3 on this hardware
  (221.1 / 689.6 vs 247.8 / 903.2). MTP first; DFlash2 is an experiment.
- vLLM #54189: ModelOpt weight-only MoE input_scale folded to zero (fixed
  by #54427). Any ModelOpt-lineage checkpoint gets the fix and a text gate.
- SlimServe build: the fork dropped `tools/build_deepgemm_C.py`; the CMake
  DeepGEMM target only triggers for arch >= 9.0, so A100 never saw the
  failure. Guarded on this branch (`cmake/external_projects/deepgemm.cmake`).
  `systemd-run --scope` does not accept `--wait`.

### SlimServe do-not-repeat list that applies here

- Fused all-reduce + mHC transition: no gain for GLM-5.3 on A100 NVLink
  (85.5 / 421.0 vs 84.0, dropped). Re-measure only with a PCIe-specific
  hypothesis, since a 10 us NVLink reduction is a 30-50 us NCCL one here.
- EP experts: slower than TP experts at every A100 config tried.
- Output-owned hidden state, channel-residual mHC, fused shared-expert
  publication, cooperative peer consumption, single-leader rendezvous,
  indexer head-sharding, top-512 radix, projection bundling: all measured
  slower on A100 (`perf/dsv4_a100_kernel_history.md:296-355`).
- Measurement anti-patterns: `--ignore-eos` TPS, synthetic repeated-token
  prompts, interval logs as benchmarks, MoE microbenchmarks with random
  `topk_ids`, sums as fingerprints.
- Acceptance is the number one confound: silent degeneration once produced
  acceptance 5.7-5.9 and forced retractions. Every spec number carries its
  acceptance and a degeneration check.
- Compile-cache A/B hazard: env-gated branches do not change the
  torch.compile cache key. Clear or key the cache for every A/B.
- Graph capture size: the 48-token verify batch once ran eager because the
  capture cap was 32 (TP8 hot c8 205 -> 465 after raising it).
- 8 GiB host tier "passed" by re-prefill; size tiers so the target survives
  LRU, and prove restores with `VLLM_KV_TIER_VERIFY=1`.

## 3b. Controls and starting point (operator decision 2026-09-04)

- External control: the rtx6kpro B12X image
  `voipmonitor/vllm:jovian-judgement-community-20260904-r24` (public, ~12 GB,
  image-only with proprietary GLM-5.3 packages) measured on this box with
  our exact-token harness and shapes. It is the fastest published stack on
  exactly this hardware, so it is the control every SlimServe number is
  read against, not just its README table.
- Secondary control: stock vLLM nightly (post #53906 + #55214) with the
  open SM120 sparse-MLA fix #55277 applied, from prebuilt wheels. Only if it
  runs; it is the "what upstream gives for free" line.
- Starting point: SlimServe's own glm53 path built for sm_120, plus the
  upstream vLLM pieces that measurably help this model ported in as
  individual changes (MTP draft model, KDA kernels, PCIe-IPC all-reduce,
  SM120 sparse MLA). The fork's vLLM base is upstream as of 2026-07-26
  (highest merged PR #49822), so these are ports of single PRs onto the
  fork, never a rebase. Once the model x quant x platform is fixed, the
  vendored QuixiCore kernels are what get tuned for this hardware.

## 3c. Build and machine constraints (standing)

- Prebuilt wheels only for torch (2.13.0+cu130) and FlashInfer
  (`flashinfer-python` + `flashinfer-cubin` + `flashinfer-jit-cache` from
  flashinfer.ai/whl, cu130). Never build FlashInfer from source or let it
  JIT a large kernel set on this box: it exhausts host RAM and crashes the
  machine (operator, 2026-09-04). The Foundry SGLang setup script is the
  proven recipe.
- The SlimServe native extension must be built (QuixiCore ops are not in
  any precompiled wheel), but capped: `CUDA_HOME=/usr/local/cuda-13.0`,
  `TORCH_CUDA_ARCH_LIST=12.0f` (one family cubin), `MAX_JOBS=8`,
  `NVCC_THREADS=2`, under `systemd-run --user --scope -p MemoryMax=120G`.
  earlyoom is active; 188 GB host RAM.
- Venvs live under `~/venvs` on the main drive, never on /raid (operator
  rule, 2026-09-04): `~/venvs/slimserve-glm53-flash`, symlinked to `.venv`.
  Scratch goes to /raid: `UV_CACHE_DIR=/raid/scratch/uv-cache`,
  `TMPDIR=/raid/scratch/slimserve-glm53/tmp` (setuptools puts the CMake
  build tree under TMPDIR, tens of GB), logs and operator scripts under
  `/raid/scratch/slimserve-glm53/`, never /tmp.
- CUTLASS via `VLLM_CUTLASS_SRC_DIR=/raid/scratch/slimserve-glm53/cutlass`
  (v4.4.2, shallow clone): the FetchContent clone hung at 0% CPU for 38
  minutes once; a local checkout removes the network from the build.
- While `uv pip install -e .` runs it holds `.venv/.lock`; any other uv
  install into that venv waits until the build finishes.
- Docker is not the house tool (operator, 2026-09-04): acceptable only when
  it is the only practical way, which the image-only B12X control is.
  Before pulling it again, containerd's root must move to /raid: Docker's
  data-root is /raid/docker but the containerd image store is
  `/var/lib/containerd` on the root disk (a 42 GB pull landed there).
- Foundry and its model servers are stopped while this work runs; the four
  GPUs are otherwise idle. The parked HunyuanImage-3 Distil weights went
  back to /raid (verified identical, duplicate removed) and
  hunyuan-v3-work moved to /raid/tmp with a symlink; root is at 65%. Attribute GPU processes by PID lineage, never by
  memory footprint (Foundry deployment notes, 2026-07-16).
- Commit authorship: one human author per commit (Eric Hartford or the
  authorized maintainer), no co-author or assistance trailers. `CLAUDE.md`
  updated on this branch.

## 3d. Port studies (2026-09-04, read-only, pre-baseline)

### PCIe-IPC all-reduce (vLLM #53576, FlashInfer #4393)
- What it is: `flashinfer.comm.PcieIpcAllReduceWorkspace`, a tuned P2P
  all-reduce for PCIe-only TP2/4/8, wrapped by upstream in a new
  `vllm/distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py`
  (239 LOC), enabled by `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`. Only
  decode-bucket shapes qualify (`[tokens, hidden]` in model dtype, numel <=
  max capture size x hidden); prefill stays on the existing path. A
  mandatory `tune()` runs at warmup before graph capture; the cache lands
  next to the FlashInfer autotune file. FlashInfer's own numbers (8x L40S,
  bf16, vs NCCL): TP4 h4096 batch 1..32 = 1.36x-2.63x. The vLLM PR body has
  no numbers (bring-up on 4x RTX 6000 Pro only).
- SlimServe today on this box: NCCL symm-mem off, FlashInfer trtllm AR off
  by default and NVLink-only anyway, custom AR disabled at world 4 without
  full NVLink unless `VLLM_CUSTOM_AR_ALLOW_PCIE=1`, symm-mem has no 12.0
  entry -> every one of the ~90 per-step reductions is `pynccl` ring
  (`cuda_communicator.py:435-449`). `CUSTOM_ALL_REDUCE_MAX_SIZES` and
  `SYMM_MEM_ALL_REDUCE_MAX_SIZES` have no "12.0" key (`all_reduce_utils.py`).
- Port surface: new file drops in unchanged (all imports exist); hand-merge
  ~35 LOC in `cuda_communicator.py` (init flags, `_initialize_communicators`,
  dispatch before `fi_ar_comm`, destroy), ~15 in `parallel_state.py`
  (`graph_capture` context, and the `destroy` order: SlimServe currently
  tears down process groups before the communicator, the bug the PR fixes),
  ~8 in `envs.py`, a warmup call in `kernel_warmup.py` before
  `capture_model`. `vllm/compilation` untouched. Roughly half a day.
- Wheel fact: the venv has NO flashinfer installed (the A100 glm53 path
  never needed it). The API is not in any tagged FlashInfer release
  (0.6.18 was cut before #4393); nightly `0.6.18.dev20260904` has
  `flashinfer_python`, `flashinfer_cubin`, and a cu130 `flashinfer_jit_cache`
  (abi3, torch-agnostic). Install the three wheels from
  https://flashinfer.ai/whl/nightly/ ; never a source build.
- Plan: Phase 1 item, off by default, after the NCCL baseline is on record.
  First microbench `benchmarks/comm/bench_pcie_ipc_all_reduce.py --hidden
  4096 --tune` from FlashInfer on the 4 GPUs (NCCL vs IPC us on this
  fabric), then one-factor A/B `..._PCIE_IPC=0/1` plus a third arm
  `VLLM_CUSTOM_AR_ALLOW_PCIE=1`, c1/c8/c16 at 1000/300, ms/step. Set
  `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` to a persistent dir. Re-run text
  gates (reduction order changes sampling; SGLang #37746). MEASURED
  2026-09-04 (torch profiler, all four ranks): the NCCL ring LL kernel is
  p50 10.3-11.1 us at c1 (8 KB) and 14-15 us at c8 (64 KB), 0.93 / 1.39 ms
  per step. The 30-50 us assumption was wrong; the PCIe-IPC ceiling is
  ~0.5 ms/step (~5% at c1), so this drops to item 5 of the Phase 1 list.
- Artifacts: /raid/scratch/slimserve-glm53/research/pr53576.diff and
  upstream_prepr/.

### MTP draft model (vLLM #53906, merged 2026-09-03)
- Upstream shape: `Glm5NextMTP` = enorm/hnorm + `eh_proj` (8192->4096) +
  `SharedHead` + one `Glm5NextDecoderLayer(is_mtp_layer=True)` (plain
  residual, NO mHC, its own DSA attention + indexer, MoE). Method is plain
  `mtp` (`glm5_next_mtp` alias in `MTPModelTypes`); draft path = target dir;
  the single layer 45 is reused for every draft step. Embeddings, lm_head
  and `shared_head.head` are shared from the target (the proposer already
  does this). The draft has no KDA layer, so no recurrent state on the
  draft side; target KDA verification uses the existing spec path (conv
  state widened by `num_spec`). DSA: with `index_share_for_mtp_iteration`
  (true in the checkpoint's text_config) step 0 computes top-k and steps
  1+ reuse it via `set_skip_topk` + `compact_topk_indices`.
- Checkpoint: 1,753 `layers.45.*` tensors in `model_mtp.safetensors`,
  names match upstream's loader; experts FP8 block [128,128] (weights
  F8_E4M3 + `weight_scale` [32,16]), everything else in layer 45 is BF16
  (attention, indexer, gate, shared experts, eh_proj are in `ignore`).
- SlimServe already has: the target skips `layers.45.*`
  (`glm5_next.py:515`), KDA conv state widened by `num_spec` (L463-476),
  proposer hooks for skip_topk/compact and head/embedding sharing
  (`llm_base_proposer.py:613-618, 1441-1533, 1584-1630`), `SharedHead`
  (`deepseek_mtp.py:47`), pooled-indexer decode sized for `1+num_spec` rows
  (`glm5_next_indexer.py:355-377, 450-461`), QuixiCore sparse MLA with
  `supports_spec_as_decode` and per-token top-k (`quixicore_mla_sparse.py`),
  FULL_DECODE_ONLY capture at query len `1+k` for UNIFORM_BATCH backends.
  Precedent profile: `qwen38fn-fp8-8` (mtp + FULL_DECODE_ONLY).
- Gaps: (1) the draft adds one MLA layer and one indexer layer, which land
  in TWO KV groups (different unified block sizes); the base proposer asserts
  one group and hands the drafter one slot mapping. Copy the
  `Qwen4ExpMTPProposer` pattern (`v1/spec_decode/qwen4_exp.py`, per-group
  block table + metadata) and extend it to slot mappings. (2) The decoder
  layer is mHC-only; add the plain-residual MTP block. (3)
  `hf_config_override` must promote `text_config` fields and carry
  `quantization_config`. (4) `model_returns_tuple` needs the new arch. (5)
  Draft MoE quant is FP8 block, not NVFP4: it must not inherit
  `moe_backend=marlin`; pick the sm_120 FP8-block MoE path via
  `speculative_config.moe_backend`. (6) Group size 11->12 changes the KDA
  padding groups (4->3) and the host-tier slab geometry. (7) The draft runs
  eager under FULL_DECODE_ONLY (perf only).
- Port order and size: `config/speculative.py` (~25) -> `registry.py` (2)
  -> `glm5_next.py` (~60: `skip_topk` plumbing, MTP block,
  `get_spec_layer_idx_from_weight_name`) -> new `glm5_next_mtp.py` (~330,
  torch RMSNorm+cat first, Triton `fused_eh_norm` later) ->
  `llm_base_proposer.py` (~5) -> new `v1/spec_decode/glm5_next_mtp.py`
  (~150) + `gpu_model_runner.py` (~40) -> profiles (~30).
- Profile: source `speculator.engine` = `{"method": "mtp",
  "num_speculative_tokens": 1, "index_share_for_mtp_iteration": true,
  "moe_backend": <sm_120 fp8-block>}`; rtx6000 variant depth 3 via
  `speculative_overrides`; keep FULL_DECODE_ONLY, capture 64, max_num_seqs
  16 (16 x 4 = 64 decode tokens). Report accepted/step and steps/s.
- sm_120 validation order: greedy k=1 equivalence (draft on vs off, same
  seed) before k=2/3; k=2,3 exercise the indexer `use_flattening` path off
  SM100; re-run the deep-context leg because group geometry changed.
- Artifact: /raid/scratch/slimserve-glm53/research/pr53906.diff (mtp.py at
  diff lines 12099-12530).

### Hybrid checkpoint inventory (RedHatAI NVFP4 vs native FP8)
- Files: /raid/scratch/slimserve-glm53/research/tensors-{redhat,native}.tsv
  (148,498 and 76,108 tensors), scripts `dump_tensors.py`,
  `classify_tensors.py`, `per_token_bytes.py` and their outputs there.
- RedHatAI backbone is all BF16 = dequant(native FP8) rounded to BF16
  (max diff one BF16 half-ulp); KDA q_proj, kv_b, lm_head, embeddings are
  bit-identical to native. Experts: per-expert `weight_packed` U8 +
  `weight_scale` F8 (g16) + F32 global scales, exactly the names
  `CompressedTensorsW4A4Nvfp4MoEMethod` expects. Total 197.8 GB.
- Native quantized (FP8 block 128 + F32 `weight_scale_inv`) only: dense
  MLP L0-2, shared experts, DSA q_a/q_b/kv_a/o_proj (+ the same in L45).
  Everything else is in `modules_to_not_convert`: the whole KDA block,
  kv_b, indexer, norms, gate, mHC, embeddings, lm_head, vision.
- Defect in the RedHatAI build: `e_score_correction_bias`, `A_log`,
  `dt_bias`, `hc_*_base/scale` were downcast F32 -> BF16. The bias sits at
  5.5-14.5 with per-layer spread 0.27-0.58, BF16 ulp 0.03-0.06 there, so
  only 6-11 distinct values survive per layer and 11,736 of 12,096 experts
  change bias rank vs native (affects `noaux_tc` top-k on every MoE
  layer). SlimServe's parameter is F32 (`deepseek_v2.py:341`); loading the
  native [288] F32 tensor fixes it. KDA decay tensors: 0.34-0.39% rel err.
- Per-token per-GPU reads at TP4 (layers 0-44 + lm_head): as shipped 4.59
  backbone + 1.19 routed experts = 5.78 GB; native-FP8 swap set 5.05;
  plus self-quantized KDA q/k/v/o 3.91; plus kv_b + lm_head 3.71.
- Loader gotcha: `Glm5NextForCausalLM` has no `hf_to_vllm_mapper` /
  `packed_modules_mapping`, so compressed-tensors `ignore`/target lists in
  HF names never match vLLM names; the backbone is unquantized only because
  no target regex hits it. Hybrid targets must be regexes on vLLM fused
  names (`fused_qkv_a_proj`, `gate_up_proj`, `in_proj_qkvgfab`) and split
  DSA vs KDA `o_proj` by layer index. FP8 block linear on 12.0 goes through
  `_POSSIBLE_FP8_BLOCK_KERNELS`; which one survives needs a GPU check.
- MTP layer 45 in RedHatAI: experts FP8 block with BF16 `weight_scale`
  (native has F32 `weight_scale_inv`), the rest BF16.

### Baseline attribution (2026-09-04, optimization_status entry of the same date)
- c1 step 9.5 ms = 8.4 ms GPU kernels in one FULL graph (~1,840 kernels)
  + ~1.1 ms host side. Physics class ~3.0 ms (KDA in_proj at 87% of
  roofline, Marlin at 63%, sparse MLA + indexer); launch-bound class ~4.1 ms
  (281 cuBLAS GEMVs 2.09 ms, mHC 0.70, ~1,070-launch tail 1.36); NCCL 0.93.
- c8 decode step ~16.7 ms = 15.1 ms GPU: Marlin 5.17 ms AT the
  expert-bandwidth floor (64 distinct experts x 3.4 MiB x 42 layers per
  rank); cuBLAS 4.3 ms for the backbone (3.8 at c1: same bytes, 56-63% of
  roofline at both); NCCL 1.39; mHC 1.13.
- Phase 1 order is therefore: (0) RedHatAI F32 tensor fix from native;
  (1) decode GEMV path for the backbone (fuse KDA f_b/g_a/g_b and DSA
  q_a/kv_a, big projections at roofline; 1.4-1.9 ms headroom); (2) launch
  tail (KDA o_norm Triton path, copies, MoE glue, norm/add fusion); (3)
  mHC transition kernel latency (grid sync + serial Sinkhorn); (4) NVFP4
  M<=16 expert decode kernel; (5) PCIe-IPC all-reduce; (6) lm_head GEMV.
  FP8 KV and the hybrid backbone bytes are the Phase 2 byte levers.

### Phase 1 item 1 pre-work: per-projection cuBLAS attribution (2026-09-04)

One decode step of the c1 and c8 traces (rank 0), cuBLAS-family kernels
grouped by (kernel, grid, block) and tied to projections by count and
launch order (perf/results/2026-09-04/glm53-nvfp4-4-rtx6000-profile/
step-gemv-c{1,8}.txt, gemv_attrib.py). Per-GPU shapes at TP4 (N x K),
bytes at bf16, roofline at 1.79 TB/s.

| projection (per GPU N x K) | per step | c1 us each | c1 % roofline | c8 us each | c8 % roofline |
|---|---:|---:|---:|---:|---:|
| KDA in_proj_qkvgfab 6288 x 4096 (51.5 MB) | 34 | 33.5 + 1.5 splitK | 82% | 34.5 | 83% |
| KDA o_proj 4096 x 2048 (16.8 MB) | 34 | 12.8 | 73% | ~9.5 (shared grid group) | ~99%* |
| KDA g_a 128 x 4096, g_b 2048 x 128, f_b 2048 x 128 (2 MB total) | 34 x 3 | 2.1 + 3.3 + 2.0 | 15% | 2.3 + 2.5 + 1.4 splitK | launch floor |
| DSA fused_qkv_a 2048 x 4096 (16.8 MB) | 11 | 12.5 | 75% | 12.3 + 1.8 splitK | 67% |
| DSA q_b 4096 x 1536, indexer wq_b 4096 x 1536 (12.6 MB each) | 11 + 11 | 10.9, 10.4 | 65% | ~9.5 | ~74% |
| DSA o_proj 4096 x 4096 (33.5 MB) | 11 | 23.3 | 80% | 21.7 | 86% |
| DSA indexer wk 128 x 4096, kpool gate 128 x 4096, weights_proj 32 x 4096 | 11 x 3 | 3.6 + 3.1 + 3.1 | launch floor | 2.5 + 1.4, 2.5 + 1.4, **21.3** | weights_proj: 2 CTAs |
| MoE router gate 288 x 4096 (2.4 MB, bf16 out then fp32 cast) | 42 | 6.4 | 21% | 3.3 + 2.3 splitK | 24% |
| MoE shared gate_up 1024 x 4096 (8.4 MB) | 42 | 7.6 | 62% | 7.3 + 4.6 splitK | 39% |
| MoE shared down 4096 x 512 (4.2 MB) | 42 | 5.1 | 46% | ~9.5 (shared grid group) | ~25%* |
| dense gate_up 6144 x 4096, down 4096 x 3072 (layers 0-2) | 3 + 3 | 33.2, 19.8 | 85%, 71% | 34.1, 21.7 | 83%, 65% |
| lm_head 38720 x 4096 (317 MB) | 1 | 199.8 | 89% | 198.4 | 89% |
| total cuBLAS family | 436 launches | 3.89 ms | 66% | 567 launches, 4.32 ms | 59% |

\* c8 groups the KDA o_proj, q_b, wq_b and shared down into one
(kernel, grid) class of 98 launches at 9.5 us mean; the split is inferred.

Reading. The big projections (in_proj, o_proj, dense, lm_head) are at
73-89% of roofline: a tuned kernel buys ~0.3 ms/step at c1 there, not
more. The waste is in the small and mid projections, which sit on the
launch floor (2-3 us per launch for <= 1 MB) or on poor cuBLAS
heuristics at M=8 (weights_proj 21 us for 256 KB, shared gate_up splitK
22 at 39%, router 24%). Headroom per step: ~1.3 ms at c1, ~1.7 ms at c8.
GateLinear's specialized tiers (cuteDSL ll_bf16, DSV3, fp32) are gated on
is_device_capability_family(100), so sm_120 falls through to tier 7
(bf16 F.linear + fp32 cast); the DSV4 A100 router kernel is fixed to
E=256/H=4096 and GLM has E=288.

Design for item 1 (one factor per A/B, in this order):
1. Fold same-input GEMVs into the existing merged linears: KDA g_a_proj
   (128 x 4096, replicated) into in_proj_qkvgfab beside f_a (a second
   replicated shard id in _KimiGDNMergedColumnParallelLinear); DSA indexer
   wk / kpool gate / weights_proj (all K=4096 on hidden_states) into
   fused_qkv_a_proj. Removes 34 + 33 launches per step; bit-exact rows.
2. Batch f_b_proj and g_b_proj (both 2048 x 128, different inputs) into one
   strided-batched GEMV launch on stacked weights. Removes 34 launches.
3. Router gate + shared-expert gate_up on the same post-norm input: one
   QuixiCore decode-projection launch over [288 + 1024] x 4096 rows with
   fp32 output for the gate rows and bf16 for the rest (generalize
   csrc/quixicore/serving/dsv4_projection_ampere.cuh: template K, M <= 16,
   dual output). Removes 42 launches, drops the fp32 cast, and takes the
   router from 6.4 us to ~2 us and gate_up from 7.6 to ~5.
4. Only then a tuned M <= 16 kernel for the 4096-row group and in_proj
   (wide loads, split-K across warps, 2-4 rows per block); microbench
   (gemv_bench.py) against cuBLAS at M = 1, 2, 4, 8, 16 first.
Expected: c1 -0.6 to -0.9 ms/step (~7-10%), c8 -1.0 to -1.3 ms (~7%).

Microbench (gemv_bench.py, weights rotated past the 128 MB L2, launches
timed inside a captured CUDA graph; gemv-bench-sm120.txt in the profile
results dir). It reproduces the trace within 5% at M=1 and settles two
questions: the practical streaming ceiling on this card is ~1.6 TB/s
(lm_head at 1.58-1.63 with either kernel, 89% of the 1.79 nominal), and
the plain row-per-block QuixiCore kernel already beats cuBLAS on every
K=4096 shape at M<=4 (in_proj 1.56 vs 1.49 TB/s, DSA o_proj 1.52 vs
1.39, fused_qkv_a 1.42 vs 1.29, router 3.0 vs 3.7 us, 128-row GEMVs 1.6
vs 2.5 us) while cuBLAS collapses at M=2 on narrow shapes (13.6 us for a
1 MB read; weights_proj 13 us at M=2..8). The row-per-block design falls
back to cuBLAS speed at M=8 on wide rows (in_proj 34.4 us) because every
block re-reads x per token from L1: the M<=16 kernel of step 4 must tile
rows per block (or use mma.sync) so x is staged once per tile. Kernel
efficiency alone is worth ~0.5 ms/step at c1; the fusions of steps 1-3
add the launch-count savings on top.

### Phase 1 progress log (2026-09-04, evening)

Retained on the rtx6000 record, one factor per boot, each with a notebook
entry (c1 / c8 / c16 at 1000/300): F32 sidecar 104.8 / 431.6 / 591 ->
o_norm Triton 110.05 / 444.6 / 603 -> g_a fold 111.0 / 445.7 / 603.9 ->
f_b+g_b batched 112.0 / 446.3 / 607.9 (+6.9% / +3.5% / +2.9% over Phase 0).
Queued and coded: router gate through the QuixiCore decode projection GEMV
(GateLinear tier 1b, fp32 logits, no cast kernel), then the indexer fold.

Item 3 scoping (mHC): the cooperative fused_pre_transition kernel is taken
only for T == 1 (py_dsv4_mhc_pre / fused_post_pre in tm_cuda_serving.cu);
every T > 1 site runs partials -> finalize_pre_mix -> apply_pre_mix, three
launches, which is the 271-launch / 1.13 ms mHC class at c8. The item is a
T <= 16 cooperative variant (grid NSPLITS x T, or T tokens per block) so
c8/c16 pay one launch per site like c1; expected 0.4-0.6 ms at c8.
Kernel builds: the CMake build tree of build 6 did not survive the
editable install, so new kernels are developed as a JIT extension
(torch.utils.cpp_extension.load) against the same csrc/quixicore source and
folded into _quixicore_C with one full rebuild at the end of the phase.

## 4. Methodology (every phase)

- Serve only through the profile: `slimserve glm53-nvfp4-4 --serve -y`
  (`--dry-run` to inspect). No hand-built commands in recorded numbers.
- Throughput: `benchmarks/benchmark_dsv4_exact.py`, `exact: true`,
  temp 1.0 / top-p 0.95 / top-k 20, seed 42, warmed, APC-hot (second of two
  identical runs). Shapes, all recorded per experiment:
  - repo canonical: 1000 in / 300 out at c1, c8, c16 (comparable with the
    A100 glm53 records) and 1K in / 2K out at c1 and c8;
  - Foundry shape: the pipeline's real director prompt (long context) at
    c1 and its 8-wide pass at c8; wall time per pass = prefill + decode;
  - lifecycle: 12K cold/hot, 128K cold/hot, post-128K continuation, one
    boot, zero preemptions.
- Speculative runs report accepted tokens per step and normalized
  `steps/s = tps / (1 + accepted/draft)`, never TPS alone.
- Correctness gates before any retained change: scoring NLL on the fixed
  text set (0.0000 boot jitter expected), needle recall leg (TP4 deep-context
  leg to 131K as the A100 record did), SHA-256 completion digests for
  bit-exact kernel changes, degeneration guard on every spec run, text and
  image requests (vision profile).
- Kernel changes: parity test against an fp32 reference (existing pattern:
  `tests/kernels/test_quixicore_sparse_mla_bf16.py`, 2e-3), microbenchmark
  at the real shapes (M = 1, 4, 8, 32, 64 per expert; real `topk_ids` from
  a captured step), then end to end. Report GB/s against 1.79 TB/s.
- Attribution: one Nsight Systems trace per phase of a c1 and a c8 decode
  step; report kernel time vs wall, launch count per token, sync count per
  step. This is the ranking input for the next phase.
- Sanity gates: TP2 vs TP4 on the same build must scale >= 1.5x
  (`CLAUDE.md`); a c8/c1 ratio under ~3.5 at NVFP4 means expert reads are
  not the limiter and something else is.
- One factor per experiment. Every retained or rejected change gets a
  notebook entry (Status / Scope / Baseline / Hypothesis / Change /
  Correctness / Results / Decision / Raw artifacts). Rejections are recorded.
- Regression rule: a change is retained only if c1 and c8 on the canonical
  shape are within noise or better, and the Foundry shape does not regress.
- Clocks: record `nvidia-smi -q -d CLOCK,POWER` per run. Stock 600 W and
  3090 MHz boost; no overclock. Optional `-lgc 3090` lock is an experiment
  with its own entry.

## 5. Phases and exit gates

### Phase 0: platform, build, baseline (no kernel work)

1. Build on tinybox per section 3c: prebuilt torch cu130 wheels into
   `~/venvs/slimserve-glm53-flash`, then the capped native build of
   `_C_stable_libtorch` for `12.0f`; smoke `import vllm._C_stable_libtorch`.
   Pull the B12X r24 control image and measure it first on the same shapes.
2. Add platform `rtx6000`: `hardware.py:_classify` branch, `profiles.json`
   platforms entry (compute capability [12,0], 96 GiB), new per-profile
   variant records for `glm53-nvfp4-4` (never widen the a100 record;
   `test_a_profile_is_one_config_per_platform` enforces it), source
   `min_gpus.rtx6000 = 4`. Host tier sized for 188 GB host RAM (not the
   A100 64 GiB per rank); NVMe tier dir is operator env.
3. Checkpoint: `RedHatAI/GLM-5.3-Flash-NVFP4` (198 GB, compressed-tensors,
   the one the profile already names and the one that produced clean text
   on 4x RTX PRO 6000 in vLLM). Download needs operator approval (966 GB
   free on /raid). Keep `/raid/weights/GLM-5.3-Flash` (native FP8) for the
   backbone in Phase 1.
4. First serve with the lowest-risk kernel set: Marlin W4A16 NVFP4 experts
   (Marlin builds for 12.0f), QuixiCore sparse NoPE MLA (check the 99 KB
   smem opt-in), Triton KDA, mHC Ampere kernels, NCCL all-reduce, BF16 KV,
   `cudagraph_mode FULL_DECODE_ONLY`, model default context.
5. Record: exact-token c1/c8/c16 on both shapes, correctness legs, the
   Phase 0 trace, and the reference bars above in `perf/baseline_status.md`.

Exit: profile healthy, correctness legs pass, baseline entry written, trace
attributes the step (bytes / sync / launch gaps / kernel inefficiency).
Expected: c1 well under 200 because of the BF16 backbone and NCCL.

### Phase 1: bytes (formats and their kernels)

1. Expert path: Marlin W4A16 is the correctness baseline on sm_120 (the
   stock CUTLASS / FlashInfer grouped FP4 GEMMs produced garbage on this
   card class, section 7). Measure Marlin at M = 1..64 per expert against
   the bandwidth floor; try the FlashInfer sm12x NVFP4 MoE (retuned #4718,
   smem fix #4013, UAF #4827 soak) only behind the SHA-digest and U+FFFD
   gates. The decode-shaped kernel in Phase 4 is the real answer for c1;
   W4A4 tensor cores are the prefill lever. Two paths by concurrency if
   the data says so (B12X pattern).
2. Hybrid checkpoint, two steps (section 3d has the tensor lists):
   (a) take the F32 small tensors from native (router
   `e_score_correction_bias`, KDA `A_log`/`dt_bias`, mHC base/scale): the
   RedHatAI build downcast them to BF16 and 97% of experts change bias
   rank, so this is a correctness fix first and is gated by NLL + needle;
   (b) take the FP8 block tensors ZAI itself quantized (dense MLP, shared
   experts, DSA q_a/q_b/kv_a/o_proj): lossless vs RedHatAI (its BF16 is
   the dequant of these), 0.72 GB/GPU saved, ceiling 310 -> ~355 at c1.
   Needs an sm_120 FP8 block linear kernel choice and regex targets on the
   fused vLLM module names (the model has no `packed_modules_mapping`).
   Self-quantizing KDA q/k/v/o to FP8 (ceiling ~460) is a separate
   experiment with its own quality gate, since ZAI kept them BF16.
   NVFP4 linear via CUTLASS W4A4 (#55170) is the follow-on A/B.
3. FP8 KV (`fp8_ds_mla`, NoPE row layout from #55277) as an explicit,
   validated per-profile choice; ~1.8x KV tokens.
4. Fix the DSV4-style launch residue that costs nothing to fix: direct_copy
   launches, dtype conversions on the decode GEMM path (exllamav3 #330).

Exit: c8 no-spec >= 740 (B12X bar), TP2/TP4 scaling >= 1.5x, correctness
legs unchanged. Notebook entries for every kernel choice including the loser.

### Phase 2: fixed overhead (the single-stream lever)

1. All-reduce: measure NCCL vs vLLM custom AR vs FlashInfer PCIe-IPC
   (#53576 port) on the 2x2 PHB topology; validate P2P copies explicitly;
   record acceptance before/after. Target <= 10 us per reduction at
   batch <= 8. Fuse residual + RMSNorm where mHC allows.
2. Launch count: full CUDA-graph coverage of the decode step including
   sparse attention (the A100 c1 diagnosis put 7-9 ms of idle in eager
   breaks), KDA projection fusion (6 -> 2 GEMMs per layer), pooled indexer
   65 -> 15 us, mHC channel ownership.
3. L2 prefetch of the next layer's backbone weights during reduction waits
   (`cp.async.bulk.prefetch.L2`), following local-inference-lab #576.
4. Trace again; kernel time should be >= 70% of wall at c1.

Exit: c1 no-spec >= 250 (vs B12X 169.9); launches per token and syncs per
step reported before/after.

### Phase 3: speculation (MTP port)

1. Port the glm5_next MTP draft path from vLLM #53906 into SlimServe's
   model and drafter infrastructure; the checkpoint ships the head
   (`model_mtp.safetensors`). Depth 3 default from the measured acceptance
   curve; adaptive depth per step; MoE-aware (draft tokens cost expert
   reads). Check hybrid page alignment (#54458) and the mixed-workload TTFT
   leg.
2. Optional: prompt-lookup n-gram first stage with MTP fallback for the
   pipeline's templated JSON. A separate entry with its own acceptance data.
3. DFlash2 stays an experiment (license, and it lost to MTP-3 here).

Exit: c1 MTP >= 400, c8 >= 1000, acceptance and degeneration guard recorded,
`steps/s` normalized.

### Phase 4: deep kernel work (sm_120-specific)

1. Decode-shaped NVFP4 expert kernel: weight-streaming grouped GEMV over
   the union of active experts, fused block-scale dequant, SiLU-mul and
   top-k weighting, >= 85% of 1.79 TB/s at M <= 64. Start from the fused
   Q4_K decode pair and cp.async ring lessons (`a100_glm52_design.md`),
   retuned for 100 KB smem and 188 SMs.
2. Fused KDA decode kernel (projections, short conv, gate, delta-rule state
   update, output norm, one launch per layer; FP32 substitution), with
   flashinfer #4709 as the reference for the state update.
3. DSA path: FlashInfer SM120 NoPE sparse MLA (#4802) vs the QuixiCore
   partitioned kernel ported to sm_120; indexer kernel at long context.
4. Persistent per-layer kernel for the decode step once 1-3 are measured.
5. Prefill: CUTLASS sm120 block-scaled GEMMs, indexer prefill sharding
   (#54951), target >= 15K tok/s at 32K and TTFT at 131K under 10 s.

Exit: c1 MTP 450-550, c8 1100-1200, prefill >= 15K tok/s; every kernel has
parity test, microbench GB/s, and an e2e entry.

### Phase 5: Foundry integration and retention

1. Foundry registry entry for the SlimServe backend (OpenAI-compatible,
   same port contract); pipeline runs at its real 8-wide fan-out with the
   concurrency cap raised to match.
2. Host + NVMe KV tier validated on this box with `VLLM_KV_TIER_VERIFY=1`.
3. Retained kernels ported to QuixiCore-CUDA after end-to-end retention.

## 6. Operator decisions

1. Checkpoint download: approved 2026-09-04 for whichever NVFP4 build the
   quant survey ranks first (section 7). Lands under /raid/weights.
2. Commit authorship: resolved 2026-09-04, see section 3c.
3. Host tier size on 188 GB RAM and the NVMe tier directory: open.
4. Foundry: stopped for the duration; the pipeline arm switches to the
   SlimServe profile when a phase beats the B12X control on the Foundry
   shape, not before.
5. Home volume at 91% (`~/venvs` 198 GB, `~/weights-parked` 158 GB): worth
   a cleanup pass before large builds land there.

## 7. Quant survey (2026-09-04)

Base: `zai-org/GLM-5.3-Flash` (MIT) is FP8-e4m3 block-128 with dynamic
activations, MTP stored as `layers.45`, BF16 twin 598.5 GiB. Every public
NVFP4 build quantizes the routed experts to NVFP4 g16; they differ in the
backbone, the MTP head, the tooling lineage, and whether anyone has seen
clean text from them on this card class.

| build | GiB | experts | backbone | MTP | quality / status |
|---|---|---|---|---|---|
| RedHatAI/GLM-5.3-Flash-NVFP4 (LLM Compressor, compressed-tensors) | 184 | L3-44 W4A4 g16; L45 FP8 | BF16 | yes, `model_mtp.safetensors` | GPQA-D 90.57, GSM8K-Plat 97.74, MATH-500 94.87; 0 U+FFFD on 4x PRO 6000; vLLM recipe default |
| local-inference-lab/GLM-5.3-Flash-NVFP4 (ModelOpt 0.39) | ~167 | W4A4 g16 calibrated; L45 MXFP8 | BF16 | BF16 shards | "KLD ~0.04"; needs the B12X fork; the control's checkpoint |
| coolbho3k/GLM-5.3-Flash-NVFP4-Optimized (RedHatAI derivative) | 175 | g16, repaired W1/W3 shared scales, tuned W2 | FP8 passthrough for dense, shared, MLA attention; KDA BF16 | absent | weight-RMSE only; unverified on sm_120 |
| tacos4me/...-NVFP4-FP8ATTN-512K (LibertAI experts + FP8 conversion) | 173 | ModelOpt NVFP4 | FP8 block incl. KDA q/k/v/o, lm_head; BF16 kv_b, gates, indexer | yes, 95.8% accept k=1 | top-1 agreement 95.57%; 144.5 tok/s TP2 on 2x PRO 6000; ModelOpt-lineage risk |
| LibertAIDAI, dealignai, RadixArk (ModelOpt) | 174-181 | NVFP4 | BF16 | yes | LibertAI and dealignai emitted 86 and 94 U+FFFD on 4x PRO 6000 (vLLM #54150); RadixArk validated only on GB300 |
| ormandj W4A16 g32 + FP8-WO backbone | 166 | g32 | FP8 block WO | source precision | GSM8K 96.9; pinned SGLang only |
| Intel AutoRound INT4 / MXFP4, AWQ (wtdcode, cyankiwi), amd Quark MXFP4 | 169-198 | INT4 or MXFP4 | BF16 | mixed | INT4 99.8% of BF16 on MMLU; MXFP4 is gfx950 only |
| EXL3 (turboderp 2-4 bpw, jmoney 4.67, K3, TR3 3.5) | 127-177 | trellis, experts only | native | native | TR3 builds are ShapleyMCG-licensed; none load in a vLLM lineage |

sm_120 kernel reality that outranks the checkpoint choice: in vLLM the
CUTLASS and FlashInfer grouped FP4 GEMMs produced garbage on RTX PRO 6000
(vLLM #54150 thread, cutlass #3096; a patched native FP4 path ran 39 tok/s
vs Marlin 46-49), so the correct stock path is Marlin W4A16 dequant. FP4 is
a bandwidth lever for decode either way; W4A4 tensor cores matter for
prefill and need our own or a fixed kernel with a correctness gate.
ModelOpt-specific bugs: weight-only MoE input_scale folded to zero
(#54189, fixed by #54427) and the unresolved U+FFFD emission (#54150).

Decision (2026-09-04): serve **RedHatAI** first (clean on this hardware,
stock loader, MTP head present, quality at parity with FP8). Downloading
to `/raid/weights/GLM-5.3-Flash-NVFP4`. Check that SlimServe's loader
picks up `model_mtp.safetensors`, which sits outside the shard index.
The control image runs its own qualified `local-inference-lab` checkpoint
(downloading into the shared HF cache at `/raid/weights/huggingface`).
Phase 1 builds the hybrid ourselves: RedHatAI experts plus the native FP8
block tensors from `zai-org/GLM-5.3-Flash` for all 45 layers, in the mixed
compressed-tensors layout RedHatAI already uses for layer 45. coolbho3k's
scale repair and tacos4me's FP8 tensor list are the references for which
tensors tolerate FP8 and which need BF16 (kv_b, KDA gates, indexer).
