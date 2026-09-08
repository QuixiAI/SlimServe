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

RTX PRO 6000 Blackwell Workstation (measured on the sm_120 box, driver 580.173.02):

| item | value |
|---|---|
| SMs / L2 / smem per SM | 188 / 128 MiB / 100 KB (99 KB opt-in per block) |
| memory | 96 GB GDDR7, 512-bit, 14 GHz -> 1.79 TB/s |
| max clocks / power | 3090 MHz SM, 600 W (stock) |
| topology | GPU0-GPU1 and GPU2-GPU3 PHB, cross-pair NODE; P2P read OK on all pairs |
| CUDA | 12.8, 13.0, 13.2 toolkits; family cubin `12.0f` needs CUDA >= 13.0 |

GLM-5.3-Flash (`<models>/GLM-5.3-Flash/config.json`, native FP8, 306 GiB):
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

| shape | today (Foundry SGLang FP8) | B12X R24 bar (4x PRO 6000) | B12X R24 on the sm_120 box, handicapped | our Phase 0 baseline | target |
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
control on the sm_120 box (handicapped)"). It is a floor we must beat in every cell,
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
- The venv lives outside the repo, symlinked to `.venv`. Scratch (the uv
  cache, `TMPDIR` for the CMake build tree, tens of GB, logs and operator
  scripts) lives outside the repo too, never /tmp; `<scratch>` below stands
  for it and `<models>` for the model cache (`SLIMSERVE_CACHE`).
- CUTLASS via `VLLM_CUTLASS_SRC_DIR=<scratch>/cutlass`
  (v4.4.2, shallow clone): the FetchContent clone hung at 0% CPU for 38
  minutes once; a local checkout removes the network from the build.
- While `uv pip install -e .` runs it holds `.venv/.lock`; any other uv
  install into that venv waits until the build finishes.
- Docker is not the house tool (operator, 2026-09-04): acceptable only when
  it is the only practical way, which the image-only B12X control is.
- Foundry and its model servers are stopped while this work runs; the four
  GPUs are otherwise idle. Attribute GPU processes by PID lineage, never by
  memory footprint.
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
- Artifacts: <scratch>/research/pr53576.diff and
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
- Artifact: <scratch>/research/pr53906.diff (mtp.py at
  diff lines 12099-12530).

### Hybrid checkpoint inventory (RedHatAI NVFP4 vs native FP8)
- Files: <scratch>/research/tensors-{redhat,native}.tsv
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
- c1 step 9.5 ms = 9.2 ms GPU kernels in one FULL graph (2,041 kernels;
  GPU-bound, ~0.3 ms outside the graph). CORRECTED 2026-09-04 evening: the
  first reading (8.4 ms + 1.1 ms host) averaged partial capture windows.
  Physics class ~3.3 ms (KDA in_proj at 82% of roofline, Marlin at 63%,
  sparse MLA + indexer); launch-bound class ~4.5 ms (309 cuBLAS GEMVs 2.30
  ms, mHC 0.77, ~1,200-launch tail 1.5); NCCL 0.99.
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
Item 2 scoping (MoE glue, from the router-gate tree's c1 trace): one MoE
layer at c1 is 17 kernels, mHC pre 8.9 us, rms 1.5, router GEMV 2.7-3.6,
shared gate_up 8.4, grouped_topk 3.7, shared silu 1.5, moe_align 2.6,
shared down 5.5, count_and_sort 1.8, fill 0.8, Marlin w13 16.1, act 1.2,
Marlin w2 10.5, moe_sum 1.3, memcpy 0.7, fused_add 0.8, NCCL 11.7. Fusable:
topk + align + count_and_sort + fill into one M<=16 routing kernel (4 -> 1
per layer, 126 launches per step), and moe_sum + memcpy + add into one or
into the Marlin w2 epilogue (3 -> 1, 84 launches). A KDA layer is already
11 kernels (mHC, rms, in_proj + splitK, fg_b bmm, conv, recurrent, memcpy,
o_norm, o_proj, NCCL). Given the in-graph gap finding, each removed
launch is worth its kernel time plus ~0.5 us of dependency latency.
Item 3 prototype (2026-09-04 18:40, JIT harness jit/qc_dev.cu): the
cooperative fused_pre_transition already indexes tokens on blockIdx.y, so a
grid of NSPLITS x T is one launch per site for any T whose blocks stay
co-resident (17 on this card at 64 splits). Parity vs the shipped paths is
summation-order level (1e-7 fp32, single-ulp bf16 flips). In-graph time per
fused post+pre site: T=1 7.5 -> 6.3 us, T=2 8.9 -> 6.5, T=4 9.0 -> 7.8,
T=8 10.2 -> 9.3, T=16 12.0 -> 13.6 (worse), so the cooperative path is
capped at T <= 8: about 0.9 us plus two launches per site at c8 (~0.2 ms
per step, ~1%), nothing at c1, nothing at c16. Production form:
step_mhc_apply.py (tm_cuda_serving.cu launcher + occupancy query), which
needs the full native rebuild; the A/B runs through the dev import hook.
Item 2 design (fused small-M routing, 2026-09-04 18:55): moe_align_block_size
takes its single-kernel small-batch mode only for num_experts <= 64 (its
shared memory is (threads+1) x experts ints), so GLM's 288 experts pay the
two-block align kernel + count_and_sort + the Python-side fill every layer,
after the separate grouped_topk kernel: 4 launches, ~9 us + 3 gaps per MoE
layer at c1. A GLM-sized kernel for M x topk <= 128 assignments does all of
it in one block: scores = sqrt(softplus(logits)) + bias from the fp32
router logits, top-8 per token, renormalize/scale, then a 288-entry count
in shared memory, warp prefix sum, block-aligned sorted_token_ids /
expert_ids / num_tokens_post_pad in the layout Marlin expects (block_size_m
8 at these M). It replaces grouped_topk 3.7 + align 2.6 + count_and_sort
1.8 + fill 0.8 us and three dependency gaps: ~0.28 ms per c1 step (~3%),
similar at c8 (M x topk = 64 assignments). Python wiring: a routing
override in the Marlin MoE path for M <= 16 that skips fused_topk_bias and
moe_align_block_size when the kernel ran. Second half of item 2 (moe_sum +
memcpy + fused_add -> one kernel) follows the same shape. Both develop in
the JIT harness and land in csrc/quixicore with the phase-end rebuild.
Item 2 prototype (2026-09-04 19:30, jit/routing.cu): one block of 512
threads does sigmoid scoring, bias-only top-8 (GLM's noaux_tc: n_group 1,
norm_topk_prob, routed_scaling_factor 2.5), and the Marlin block alignment.
Against the router's own grouped_topk and moe_align_block_size: ids
identical, weights within 9e-8, block layout identical, M = 1..16. In-graph
per layer: 6.3 -> 3.7 us at M=1, 6.8 -> 4.8 at M=8, 6.8 -> 6.0 at M=16,
and three launches fewer (~0.15-0.2 ms per c1 step with their gaps).
Plumbing next: GroupedTopKRouter override for M <= 16 that also returns the
alignment, and fused_marlin_moe accepting it instead of recomputing.
Kernel builds: the CMake build tree of build 6 did not survive the
editable install, so new kernels are developed as a JIT extension
(torch.utils.cpp_extension.load) against the same csrc/quixicore source and
folded into _quixicore_C with one full rebuild at the end of the phase.

Illegal memory accesses of 2026-09-04/05, root cause found 2026-09-07 (notebook
entry of that date): the fused routing kernel was not total over NaN logits.
vLLM's dummy runs (capture warmups, sampler warmup) put NaN activations
through the router on every boot; the kernel selected "expert 288", aliased
its shared counters and laid assignments after padding, and Marlin, which
takes the first num_valid entries of a block as tokens, read the padding
value as a row one past the activations. Found with a GPU core dump
(cuda-gdb: Marlin, Warp Illegal Address, the block's shared ids, A all NaN,
A row prob_m unmapped) after the SASS of all three builds proved identical.
Fixed in the kernel (NaN ranks below finite, selected experts -inf, shared
sel), tested on non-finite inputs, native rebuild the same day. The
2026-09-04 rejection of the cooperative mHC launcher rested on faults of the
same signature and is unproven until re-tested on the fixed build.

Item 2 remainder, done 2026-09-07 afternoon (notebook entry of that date):
the modular kernel now hands Marlin its own output buffer when the finalize
step is a no-op (the copy after every MoE layer is gone), and the runner
folds the shared-expert add into the Marlin sum with the QuixiCore
moe_sum_add kernel (shared experts launched ahead of the routed experts on
the aux stream, joined where the sum consumes them, one launch instead of
moe_sum + copy + add). 1379 -> 1298 launches per decode step, -0.5 us per
MoE layer tail, step time -0.9% in the profiler pair (inside boot spread),
exact-token unchanged to +0.7%. Two side fixes came with it: the
functionalization pass handles the single-output moe_forward op, and the
compile cache key carries the QuixiCore graph capabilities. The kernel
binding ships with the next native rebuild; until then the Python path is
inert (old op, old kernels).

Item 4, done 2026-09-07 (notebook entry of that date): a bf16 M <= 16
tensor-core GEMM (m16n8k16 mma.sync, cp.async-staged K chunks, warps
splitting K, one shared-memory reduction) replaces cuBLAS on the unquantized
backbone projections with 2048..16384 rows and K a multiple of 128 - KDA
in_proj and o_proj, the DSA projections, the dense MLP, the shared-expert
down - through an opaque custom op whose M branch stays inside the op, so
torch.compile traces one graph. Kernel-busy time -0.20 ms (-2.4%) per c1
step, 34 split-K reduce launches gone, every replaced shape at or below
cuBLAS at every M; +1.5% c1 / +1.3% c8 / +0.7% c16 between like boot states,
gates in band. Kill switch SLIMSERVE_DECODE_GEMM=0. Left on cuBLAS on
purpose: lm_head (parity), shared gate_up 1024 rows (loses at M=1), the
K=128 KDA projections. The measurement itself produced the boot-spread
finding (same notebook date): six boots of identical code span 4-5% at c1
with identical per-kernel times. The probes of the same evening (notebook
"Probe results") ruled out the CPU governor, GPU clocks and host-thread
placement and located it: per-node latency inside the CUDA-graph replay,
set once per boot and the same on all four GPUs (in-step idle 0.37 vs
0.68 ms over ~1250 nodes), not reproducible on one GPU in isolation. The
profiler's kernel-busy time (overlap-free) is the decision metric for
kernel work, every exact-token boot can label its own state (ab.sh
STATE=1: a profiled round after the benches), and exact-token arms are
compared like-state.

Item 2b, done 2026-09-07 (notebook "FP8 weight swap-set" parts 1 and 2):
the native checkpoint's block-FP8 tensors (e4m3, 128x128 fp32 scales) for
the dense MLP, the shared experts and the DSA q_b / o_proj are served in
place of their RedHatAI BF16 twins through a 2.53 GB sidecar
(`slimserve/fp8_swapset.py` builds and verifies it, bit-exact against the
BF16 tensors; the loader substitutes 157 weights and injects 157 block
scales; a compressed-tensors group `slimserve_fp8_swapset` is merged from
the manifest and hashed into the compile cache). Decode runs a W8A16
variant of the item 4 GEMM (`csrc/quixicore/serving/fp8_decode_gemm.cuh`,
dequant with the block scale in the load path, the mma multiplies exactly
the BF16 values served before), prefill the compiled sm_120 CUTLASS
blockwise kernel. fused_qkv_a stays BF16 (its indexer shards are BF16 in
native). Like-state profiler pair: wall per c1 step 7.29 -> 6.94 ms
(-4.8%), overlap-free busy -0.35 ms; exact-token fast-state boots c1 117.4
-> 123.8, c8 469 -> 478, c16 618 -> 628 (+5.5 / +2.0 / +1.5%), gates in
band, every bench exact. Kill switches SLIMSERVE_FP8_SWAPSET=0 and
SLIMSERVE_DECODE_GEMM_FP8=0. Open: the small-K fp8 shapes on the aux
stream (shared down K=512 8.7 us vs bf16 4.0 under Marlin contention) -
hidden today, a sidecar variant without shared down or a K<=1536 kernel
variant is the one-factor follow-up; the C++ binding rides the next native
rebuild (JIT hook until then).

Phase 2 first lever, done 2026-09-07 (notebook "Custom all-reduce"):
vLLM's custom all-reduce enabled on the PCIe-only TP4 topology through
the fork's VLLM_CUSTOM_AR_ALLOW_PCIE=1 escape hatch, now the rtx6000
record's env default. 90 x 5.1 us cross_device_reduce kernels on the
compute stream replace 91 x 11.2 us NCCL ring-LL launches and their
cross-stream graph gaps; c1 123.8 -> 137.4, c8 478 -> 499, c16 628 ->
657 on the swap-set tree, six gates in band. Record then 137.4 / 499 /
657 no-spec.

Phase 2 second lever, done 2026-09-07 21:10 (notebook "KDA projections
self-quantized to block FP8"): the 34 KDA layers' q/k/v/beta/f_a/g_a and
o_proj, BF16 in every checkpoint, quantized here to the swap-set's
128x128 block-e4m3 format (absmax / 448) and served through the same
W8A16 decode GEMM; per rank in_proj 34.3 -> 18.3 us, o_proj 11.7 -> 6.9,
union busy -0.65 ms/step. The merged projection's 16-row beta shard is
stored per rank padded to a 128-row block (TP-specific sidecar, manifest
`tp_size`, loader guard) so every shard's block scales load through the
standard merged loader. Quality cost measured and accepted under this
plan's own tolerance: -0.014 nats mean NLL over 8 vs 26 readings, needle
unchanged. Record now 152.7 / 519 / 679 no-spec (fast state); the c8 bar
(740) is 70% covered. Revert = relink the plain sidecar.

### Bytes research digest (2026-09-07, read-only; full notes in <scratch>/research/fp8-{swapset,kv}-research-2026-09-07.md)

FP8 weight swap-set (Phase 1 item 2b): the native checkpoint's FP8 block
tensors (e4m3, 128x128 F32 scales) cover exactly the dense MLP (L0-2), the
shared experts (L3-45) and the DSA q_a / kv_a / q_b / o_proj; every one has
a bit-identical-name BF16 twin in the RedHatAI checkpoint. Per GPU per
token at TP4 the swap saves 0.723 GB (1.447 -> 0.724 GB; step 5.78 ->
5.05 GB), or 0.631 GB if fused_qkv_a_proj stays BF16 (its three indexer
shards are BF16 in native, and one module takes one scheme). The RedHatAI
config is already compressed-tensors "mixed-precision" with an FP8-block
group for the MTP experts, so a third group with targets on the fused vLLM
names (DSA layers listed explicitly: KDA layers also call their output
projection o_proj) is all the config needs; the loader needs the F32
sidecar generalized to inject weight_scale tensors (renamed from
weight_scale_inv, F32). Kernel: on sm_120 the block-FP8 linear resolves to
CutlassFp8BlockScaledMMKernel (compiled in; swapAB path for M <= 64; no
test in the tree; +1 activation-quant launch per linear = +134/step) with
DeepGEMM silently outranking it if that package ever lands (pin it). The
alternative that keeps one launch and bf16 activations is a W8A16 variant
of the item 4 decode GEMM (dequant with the block scale in the load path;
K chunk 128 = scale block). Evidence first: parity + microbench of both at
M = 1..16 on the six swap shapes, then the sidecar + config + gates.

FP8 KV (Phase 1 item 3): no path exists for this backend (the fp8 kernels
assert GLM-5.2's 576-wide q; the cache-write kernel requires pe_dim 64;
QuixiCore has no inline per-128 scale mode; the indexer cache is bf16
only). At the benchmarked 1000-token shape the DSA path is dense (512
pools selectable, 250 present) and reads 16.9 MB/step at c1, so FP8 rows
save at most 0.15 ms/step (1.6%); it is a capacity lever (1.45x with
fp8_ds_mla NoPE rows, 1.93x with the indexer row too; a plain 512 B row
with the existing per-tensor scale mode gives 1.50x with today's page
layout) and raising gpu_memory_utilization already gives 1.40x for free.
Deprioritized behind the swap-set; returns as a long-context capacity item
once the NLL / needle legs run on this platform.

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
- Kernel choice for anything on the aux stream (2026-09-07, swap-set part
  3): a kernel that runs underneath the Marlin routed-expert kernels is
  tuned from the serving trace under contention (aux_window.py around one
  launch), never from the isolated bench alone. The shared-expert gate_up
  config the bench preferred (8 rows / 4 stages, 4.5 us alone) took 21 us
  next to Marlin where 8 stages took 7; the decision metric stays the
  overlap-free busy of a state-labelled boot.
- Boot-state label on every exact-token boot (ab.sh STATE=1): the profiled
  round after the benches gives gap_locate.py's in-step idle (the fast /
  slow front-end state) and step_attrib's kernel-busy; arms are compared
  like-state, and pass 2 of c1 is the quoted number (pass 1 can dip 4% in
  some boots).
- Attribution: one Nsight Systems trace per phase of a c1 and a c8 decode
  step; report kernel time vs wall, launch count per token, sync count per
  step. This is the ranking input for the next phase.
- Sanity gates: TP2 vs TP4 on the same build must scale >= 1.5x
  (`CLAUDE.md`); a c8/c1 ratio under ~3.5 at NVFP4 means expert reads are
  not the limiter and something else is.
- One factor per experiment. Every retained or rejected change gets a
  notebook entry (Status / Scope / Baseline / Hypothesis / Change /
  Correctness / Results / Decision / Raw artifacts). Rejections are recorded.
- Instrument for launch-count changes (2026-09-04, evening): decide on a
  same-tree profiler pair (perf/results/.../route-fused-profile/prof_pair.sh:
  arm A then arm B booted back to back, one c1 capture each, full decode
  steps only, compare GPU span and launches per step), not on the
  exact-token harness, whose boot spread (below) hides anything under ~3%.
  The spread itself is inter-kernel idle inside the CUDA graph (0.02 vs
  0.46 ms per step on the same tree), on the GPU timeline, not host work.
- Noise band on this stack (measured 2026-09-04, both sessions): the
  exact-token c1 cell moves up to ~4% between boots of the same tree
  (pass-to-pass within a boot <0.3%); the 8-slice prompt_logprobs mean
  moves 0.02-0.03 nats between boots and per-token logprobs have sd 0.47
  nats even between two requests on one boot (MoE routing flips from
  nondeterministic reductions). Rules: a throughput delta under 3% needs
  two boots per arm; discard a c1 cell that starts in the same second as
  /health; scoring gates compare boot means with a 0.03 nat band; there
  is no bit-exact greedy gate on GLM-5.3 here, parity tests carry that
  burden for kernel changes.
- Regression rule: a change is retained only if c1 and c8 on the canonical
  shape are within noise or better, and the Foundry shape does not regress.
- Clocks: record `nvidia-smi -q -d CLOCK,POWER` per run. Stock 600 W and
  3090 MHz boost; no overclock. Optional `-lgc 3090` lock is an experiment
  with its own entry.

- Non-finite inputs (2026-09-07): a kernel that emits indices or layouts
  (routing, alignment, top-k, gather tables) is tested on NaN and inf inputs
  before it is retained, because every boot runs the model on dummy
  activations that vLLM itself documents as possibly inf/NaN
  (`_dummy_sampler_run`). The output must be a layout the consumer can take
  for any input; "garbage in, garbage out" is acceptable, out of range is
  not. The reference kernels (grouped_topk, moe_align_block_size) meet this.
- Illegal-memory-access triage (2026-09-07): do not reason from boot
  conditions; the fault surfaced as "profiler boots" only by allocator
  layout luck. Read `dmesg` for the `Xid 31` lines first (faulting address,
  PID -> rank, page alignment; one signature across several faults means one
  cause), then reproduce with `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
  CUDA_COREDUMP_FILE=<dir>/core_%h_%p.nvcudmp` and read the dump with
  cuda-gdb (`target cudacore`, `info cuda kernels`, `x/i $pc`,
  `x/8xw (@shared int*)`, `x/8xw (@global int*)`): kernel, exception type,
  the block's shared state and whether the accessed page is mapped. A full
  dump is the context size (87 GB, ~15 min) and the process must not be
  killed while it is written. Compare SASS across builds (cuobjdump) before
  suspecting the compiler.

## 5. Phases and exit gates

### Phase 0: platform, build, baseline (no kernel work)

1. Build per section 3c: prebuilt torch cu130 wheels into the venv, then the capped native build of
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
   on 4x RTX PRO 6000 in vLLM). Download needs operator approval. Keep `<models>/GLM-5.3-Flash` (native FP8) for the
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
   experiment with its own quality gate, since ZAI kept them BF16 - DONE
   2026-09-07, retained at -0.014 nats (Phase 2 second lever, below).
   NVFP4 linear via CUTLASS W4A4 (#55170) is the follow-on A/B.
3. FP8 KV (`fp8_ds_mla`, NoPE row layout from #55277) as an explicit,
   validated per-profile choice; ~1.8x KV tokens.
4. Fix the DSV4-style launch residue that costs nothing to fix: direct_copy
   launches, dtype conversions on the decode GEMM path (exllamav3 #330).

Exit: c8 no-spec >= 740 (B12X bar), TP2/TP4 scaling >= 1.5x, correctness
legs unchanged. Notebook entries for every kernel choice including the loser.

Phase 2 third lever, done 2026-09-07 22:06 (notebook "Small-k sampler
kernel (QuixiCore topk_sample)"): the decode sampler's fused Triton
top-k/top-p kernel plus full-vocabulary softmax (~180 us per step at every
concurrency) replaced by two QuixiCore launches (radix top-32 candidates
per row, then merge + masks + softmax + noise argmax on the candidates)
when every request's top_k <= 32; sampling class 0.17 -> 0.01 ms/step,
like-state +2.8% c1 / +1.0% c8 / +1.0% c16, gates in band. Same session:
mHC cooperative launch extended to T <= 8 by default (+1.2% c8), custom AR
forced one-stage rejected (-1.5% c8 / -2.6% c16), c16 attribution on the
final tree (Marlin 52% at the expert-bandwidth floor; the movable
remainder is the mHC split path at T = 16, the reduces, the logits
all-gather and ~1000 sub-5 us launches). Fast/fast-state throughput of
this tree not yet observed; like-state predicts ~157 / 525 / 685.

Phase 2 fourth lever, done 2026-09-08 09:42 (notebook "Sparse MLA decode:
partition and reduce"): the DSA layers' sparse decode reduce switched to
the one-thread-per-channel reducer and the index-list partition set by
batch (32 tokens at B <= 8, 64 above): 0.35 -> 0.13 ms/step at c1.
Record now (fast/fast state, no-spec, 1000/300): 162.8 / 534.5 / 691.0
tok/s; the c8 bar (740) was 72% covered. With the NCCL P2P env (below)
the fast/fast record was 163.1 / 556.1 / 728.9 (p2p-rec2, 2026-09-08
10:26); with the mHC prefill kernel (below) it was 164.7 / 575.7 / 764.8
(mhcpf-rec1, 10:52); with the pooled-indexer prefill path (below) it is
165.2 / 585.1 / 780.7 (idx-rec3, 11:51); with the sparse MLA prefill
kernel (below) it is **165.7 / 591.9 / 797.4** (mlapf-rec4, 12:29): the
c8 bar (740) is 80% covered.

Prefill, found 2026-09-07 22:40 (notebook "Prefill attribution at c8"):
the bars are measured on 1000-in / 300-out requests whose prefill is
18-20% of the c8/c16 wall at 8k tok/s, two thirds of it the TP all-reduce
over PCIe (57 MB per site through NCCL's LL ring at 15 GB/s) and an mHC
partials kernel shaped for decode (7% of HBM bandwidth). Prefill levers
now sit in this plan next to the decode ones: large-message reduce path
(NCCL protocol, custom AR cap), an mHC prefill kernel, FP4 tensor-core
grouped GEMM at M >= 64, prefill attention for the sparse MLA path. The
c8 bar (740) needs prefill + 300 x step <= 3.24 s per request: with
prefill halved the decode step must reach 9.4 ms (12.1 today).

Prefill first lever, done 2026-09-08 10:11 (notebook "Prefill: the
all-reduce arms"): the large-message reduce now runs over PCIe P2P in NCCL
(record env NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS; the box exported
NCCL_P2P_DISABLE=1 and NCCL's default P2P level refuses the PHB/NODE
topology, so it was on host-memory SHM). c8 prefill 811-831 -> 666-678 ms,
c16 1354-1361 -> 1103-1106 (-18%); aggregates +4.2% c8 / +5-6% c16 in a
slow-state boot; decode untouched. Raising the custom all-reduce cap
(64/128 MiB, new knob VLLM_CUSTOM_AR_MAX_SIZE_MB) gave -14% and lost to it,
so the cap stays 8 MiB. Profile env is setdefault: an operator export of
NCCL_P2P_DISABLE=1 shadows the record (the CLI plan print now says so).
Every VLLM_* value is a torch-compile cache-key factor with the Triton
cache inside that directory: a new value costs one longer boot and a
~49 s first request (12 KDA-prefill/indexer kernels outside the boot
warm-up), a profile follow-up.

Prefill second lever, done 2026-09-08 10:50 (notebook "mHC prefill
partials: a kernel shaped for T = 7000"): the split path's `partials<24>`
(20% of the prefill step, 2.06 ms per launch at 7001 tokens, 7% of HBM)
replaced above T = 64 by `partials_prefill`: a block per 128-dim slice of
all four streams and 32-token tile, fn and residual staged once, whole
dot products per lane; 0.39 ms per launch at 1.3 TB/s, fused residual
bit-exact. Served: c8 prefill 644-678 -> 532-549 ms, c16 1101-1106 ->
873-875; aggregates +4% c8 / +5% c16 like-for-like, decode identical,
gates in band. Remaining prefill levers: FP4 grouped GEMM at M >= 64
(Marlin at ~25% of the tensor-core rate), the sparse MLA prefill walk,
the pooled indexer, the fp8 blockwise GEMM; re-attribution queued.

Prefill third lever, done 2026-09-08 11:37 (notebook "Pooled indexer
prefill: pooled keys once per request, tiled tensor-core scoring"): the
DSA indexer's prefill rows no longer re-pool every visible pool per row
(52 ms of the 599 ms c8 step); pooled keys are built once per request and
scored as a tiled tensor-core matmul. Served, same-tree control: c8
prefill 532 -> 487 ms (-8.4%), c16 876 -> 799 (-8.8%), gates in band. Two
correctness bugs found by its parity test were fixed first (a tail-store
race in the top-k expansion that dropped the query's own token from 1-2%
of rows, and a KV-indexed row -> request map that was wrong after a
prefix-cached request). Remaining prefill levers: the sparse MLA prefill
walk (63 ms on the decode kernel), the fp8 blockwise GEMM (61), Marlin
at M >= 64 (112, needs a purpose-built FP4 grouped GEMM: the CUTLASS
sm120 backend as wired lost 7% prefill and 13-25% decode).

Prefill fourth lever, done 2026-09-08 12:06 (notebook "Sparse MLA
prefill: head-batched tensor-core attention over the top-k list"): the
DSA layers' sparse attention over each token's top-k list ran prefill
chunks on the decode walk (one warp per head and token, CUDA-core dot
products; 63 ms of the c8 step). A Triton kernel with one program per
token and the heads as the tensor-core M dimension replaces it for steps
with prefill tokens: 11.5 -> 1.7 ms per call isolated; served c8 prefill
489 -> 446 ms (-9%), c16 802 -> 730 (-9%), gates in band, decode
untouched. Remaining prefill levers: the fp8 blockwise GEMM (61 ms), the
FP4 grouped GEMM (Marlin 112 ms), and the reduce at the wire (168 ms,
physics: micro-batch overlap is the only lever left there).

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
   quant survey ranks first (section 7). Lands under the model cache.
2. Commit authorship: resolved 2026-09-04, see section 3c.
3. Host tier size on 188 GB RAM and the NVMe tier directory: open.
4. Foundry: stopped for the duration; the pipeline arm switches to the
   SlimServe profile when a phase beats the B12X control on the Foundry
   shape, not before.

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
to `<models>/GLM-5.3-Flash-NVFP4`. Check that SlimServe's loader
picks up `model_mtp.safetensors`, which sits outside the shard index.
The control image runs its own qualified `local-inference-lab` checkpoint
(downloading into the shared HF cache at `<models>/huggingface`).
Phase 1 builds the hybrid ourselves: RedHatAI experts plus the native FP8
block tensors from `zai-org/GLM-5.3-Flash` for all 45 layers, in the mixed
compressed-tensors layout RedHatAI already uses for layer 45. coolbho3k's
scale repair and tacos4me's FP8 tensor list are the references for which
tensors tolerate FP8 and which need BF16 (kv_b, KDA gates, indexer).
