# GLM-5.3-Flash NVFP4 on 4x RTX PRO 6000 Blackwell (sm_120): campaign handoff

Status: DRAFT FOR REVIEW, 2026-09-11. Branch `glm53f-rtx6000`, cut from
`upstream/main` 2b355117e (QuixiAI/SlimServe, 2026-09-11). No code changes yet.
This document is the regimen, the rubric and the roadmap for the campaign. The
running notebook is `perf/optimization_status.md`, record rows go in
`perf/baseline_status.md`, raw logs under `perf/results/YYYY-MM-DD/<run-id>/`.

Owner of the machine and the decisions: the operator (auroter). Commit identity
for this branch: `Auroter <auroter@users.noreply.github.com>`, no co-author or
assistance trailers, no mention of automated assistance (CLAUDE.md names Eric
Hartford as the sole author for his own commits; on this box the operator's
identity is used, per the operator's 2026-09-09 instruction).

## 0. Mission

Serve **GLM-5.3-Flash** from **`RedHatAI/GLM-5.3-Flash-NVFP4`** (NVFP4 group-16
routed experts, FP8 tail layer, BF16 backbone as shipped) on **tinybox** (4x
RTX PRO 6000 Blackwell Workstation, sm_120, PCIe 5 x16, no NVLink, driver
580.173.02 = CUDA 13.0) through a new `rtx6000` record of the registry profile
`glm53f-nvfp4-4`, and make that record the fastest way to run this model,
quant and card combination anywhere: decode (single-stream and concurrent) and
prefill both measured, both beating the published external stacks, with the
QuixiCore kernels tuned for exactly this hardware.

The operator's definition of the work (2026-09-11): research-led kernel
optimization, prioritized by expected end-to-end gain, tested only to validate
that a change helps, journaled the way the earlier campaigns were journaled,
carried as far as it practically goes without diminishing-returns sweeps or
validation-for-its-own-sake, and finished with a reviewable PR.

"Done" means: a Phase 6 PR against `QuixiAI/SlimServe` main that carries the
`rtx6000` record, its kernels, their tests, and notebook entries with the
retained numbers; every retained kernel ported to QuixiCore-CUDA; and a
baseline row that beats the local B12X control in every cell on our own
protocol and the physics-derived targets in section 4 where the notebook
proves they are reachable.

## 1. The previous campaign (Codex, 2026-09-04 to 2026-09-11): what to keep

The prior agent ran on the deleted branch `glm53-flash-sm120` (277 commits
past upstream/main, 365 files, 83K lines added). Its five closed review PRs
(#24 to #28 on QuixiAI/SlimServe) are still fetchable as
`refs/pull/{24,25,26,27,28}/head`; locally they are `upstream/pr/24` (tip
24561e742) and `upstream/pr/26` (kernels). The early state (Sep 4, 28 commits)
survives as the local branch `glm53-flash-sm120-mtp`. Nothing from it is on
this branch.

### 1a. What it measured (rtx6000 record, no speculation, exact-token 1000/300, aggregate output tok/s c1 / c8 / c16)

| step | change | c1 | c8 | c16 | how measured |
|---|---|---:|---:|---:|---|
| Phase 0 | a100 kernel set built for 12.0f; BF16 backbone, Marlin W4A16 experts, NCCL ring AR | 104.8 | 431.4 | 591.0 | two passes, one boot |
| 1 | F32 sidecar (native F32 router bias, KDA A_log/dt_bias, mHC base/scale; RedHatAI downcast them) | neutral | neutral | neutral | quality +0.03 nats |
| 2 | KDA o_norm through the Triton kernel instead of the decomposed forward | 110.1 | 444.6 | 603.0 | +5.0 / +3.0 / +2.0 % |
| 3 | g_a folded into in_proj; f_b + g_b as one strided bmm | 112.0 | 446.3 | 607.9 | launch count |
| 4 | router gate through a QuixiCore bf16-in/fp32-out GEMV; indexer wk/gate/weights_proj folded into fused_qkv_a | 113.3 | 456.3 | 609.9 | +1.0 / +1.3 % |
| 5 | fused small-M routing + Marlin alignment (`glm_route_align`), Marlin workspace once per device, `moe_sum_add` combine | ~113 | ~456 | ~610 | profiler pair: -81 -37 -80 launches, -2 % step |
| 6 | bf16 M<=16 tensor-core decode GEMM on backbone projections (in place of cuBLAS) | 117.4 | 467.4 | 617.5 | +1.5 / +1.3 / +0.7 % like-state |
| 7 | FP8 swap-set: ZAI's native e4m3 block-128 tensors for dense MLP, shared experts, DSA q_b/o_proj, on a W8A16 decode GEMM | 123.8 | 478.2 | 627.5 | +5.5 / +2.0 / +1.1 % |
| 8 | vLLM custom all-reduce over PCIe P2P (`VLLM_CUSTOM_AR_ALLOW_PCIE=1`; 5.1 us vs NCCL 11.2 us per reduction) | 137.4 | 499 | 657 | +11.0 / +4.4 / +4.7 % |
| 9 | self-quantized FP8 KDA in_proj/o_proj (ZAI kept them BF16; NLL gate passed at -0.014 nats) | ~152 | ~519 | ~679 | +10.6 / +4.5 / +2.6 % |
| 10 | small-k sampler fix; mHC cooperative launch for T<=8 as default | ~157 | ~531 | ~685 | predicted from pairs |
| 11 | sparse MLA decode: 32/64-token partitions + one-thread-per-channel reducer | **162.8** | **534.5** | **691.0** | +3.7 / +0.7 / +0.9 %; best fast-state boot |
| 12 | custom AR max size 8 -> 64 MiB (prefill chunks stop falling back to NCCL) | 162.8 | 550.4 | 698.0 | c8 prefill -14 % |
| final | merged-tree cold-prefix protocol (E2E incl. prefill, three repeats) | 156.8 | 577.5 | 781.7 | different protocol; not comparable to the rows above |

Prefill: TP-row-sharded indexer prefill (from vLLM #54951) cut cold 128K
engine TTFT from 10.66 s to 10.01 s; the H16 BF16 swapAB sparse prefill took
another 0.3 to 0.5 %. Cold 32K TTFT ended at ~2.5 s.

Rejected with evidence (do not repeat without a new hypothesis): cooperative
mHC for T<=8 (neutral), output-parallel mHC partials, mHC + RMSNorm fusion,
tensor-core mHC (quality gate), 8 MiB next-weight prefetch during the KDA
output (c8/c16 -4.4 %), whole-head KDA conv/state/norm fusion (regressed),
fused radix pool ordering (latency), a decode-shaped NVFP4 expert GEMM
microbench that lost to Marlin, MTP k=3 as the default (Foundry 8-wide
c8 -20 %, prose exact-token -15 to -30 %), DFlash2 in the B12X image (lost to
MTP-3 there), shared-expert down back to BF16 (neutral).

### 1b. What went wrong

Every throughput gain above landed between the evening of Sep 4 and the
morning of Sep 8. From Sep 8 on, the notebook grew by roughly 100 entries about
reproducibility audits, sampler tie semantics, quality-score variation
root-causing (MoE reduction order, indexer top-k ordering, layer-23 cutoff
ties), deterministic-reduction diagnostics, prompt-score chunking, graph
observers, B12X control re-runs, and then three days of PR splitting and
CodeRabbit review. Throughput did not move after Sep 8 (162.8 -> 156.8 on a
stricter protocol). The operator deleted the branch for that reason. The
rules in section 6 exist so it does not happen again.

### 1c. What survives on disk and is safe to reuse

- Checkpoints: `/raid/weights/GLM-5.3-Flash-NVFP4` (RedHatAI, 191 GB, with
  `model_mtp.safetensors`), `/raid/weights/GLM-5.3-Flash` (ZAI native FP8,
  306 GB), and the sidecars next to the RedHatAI dir:
  `f32-overrides.safetensors` (1.2 MB) and `fp8-swapset.{safetensors,json}`
  (7.2 GB, 157 native FP8 weights + scales). `/raid/weights/
  GLM-5.3-Flash-NVFP4-FP8-KDA-TP4/` is the symlinked recipe dir for the FP8-KDA
  variant (recipe id `glm53-redhatai-nvfp4-fp8-kda-tp4-v1`). These are data
  artifacts, verified bit-exact at build time; reusing them costs nothing.
- Scratch harness under `~/.local/scratch/slimserve-glm53/`: `serve.sh`
  (profile launcher under a 150G memory scope), `bench.sh` (exact-token
  shapes), `ab.sh` (boot, two passes, gates, state label, stop), `gate.py`
  (NLL on 8 fixed slices + needle margins), `prof_pair.sh` + `prof_run.py` +
  `step_attrib.py` + `gap_locate.py` (same-tree profiler pairs), `rebuild.sh`
  and `native_incremental.sh` (the 12.0f build recipes that worked),
  `post_rebuild_validate.sh`, `prompt-source.txt`, the CUTLASS v4.4.2
  checkout, and the B12X control drivers (`control.sh`, `control-bench.sh`).
  The scripts refer to the old profile name `glm53-nvfp4-4` and to harness
  flags that upstream's `benchmark_dsv4_exact.py` does not have; Phase 0
  adapts them, it does not rewrite them.
- Docker image `voipmonitor/vllm:jovian-judgement-community-20260904-r24`
  (the B12X control) is pulled.
- Venv `~/venvs/slimserve-glm53-flash` (torch 2.13.0+cu130) is linked as
  `.venv`. The `.so` files in `vllm/` were built from the deleted branch's
  `csrc` and are STALE for this tree: rebuild before any serving.

### 1d. Salvage verdict (code read 2026-09-11)

Operator's answer: "if there's kernel pieces that work then great, we can
use them; I don't care about throwing away code." The five closed PRs added
~83K lines; the serving-path code worth carrying is about 1,500. The four
kernel headers were read in full: they are small, clean and built like the
repo's own Ampere kernels (cp.async staging, ldmatrix, mma.sync m16n8k16,
one shared-memory reduction, total over bad inputs).

KEEP, re-landed one commit at a time and re-measured on this tree:

| piece | lines | verdict |
|---|---:|---|
| `bf16_decode_gemm.cuh` | 234 | weight-streaming M<=16 tensor-core GEMM; +1.5 % alone, and the base of the FP8 kernel |
| `fp8_decode_gemm.cuh` | 212 | same pipeline, exact e4m3 -> f16x2 conversion, one 128x128 scale per K chunk; W8A16 output bit-identical to a BF16 dequant; the kernel behind +5.5 % and +10.6 % c1 |
| `glm_moe_routing.cuh` | 173 | fused scoring / top-8 / renorm / Marlin alignment in one block, NaN-total; drop the `STABLE_ALIGNMENT` template branch (a determinism diagnostic) |
| `glm_moe_combine.cuh` | 80 | fused routed sum + shared-expert add |
| decode-linear custom op in `layers/utils.py` | ~180 | correct compile-opaque pattern with the compile-cache factor |
| `fp8_swapset.py`, `f32_overrides.py` builders + the `glm5_next.py` loader hooks | ~600 | keep; the KDA self-quant bakes a TP4-specific beta-shard layout into the artifact (fine for a per-platform record; note it in the record) |
| indexer fold (wk / kpool gate / weights_proj into `fused_qkv_a_proj`) | +71/-23 | not upstream; re-apply from the exported patch |
| Marlin workspace allocated once per device | ~15 | upstream still allocates per call; re-implement |
| sparse MLA partition 32 (B<=8) / 64 rule + channel reducer selection | ~30 | upstream's `_bf16_partition` is a 128-token scratch-cap rule and no binding uses `paged_attention_reduce_channels`; re-implement from the notebook |
| tests `test_quixicore_decode_gemm{,_fp8}.py`, `test_quixicore_glm_route_align.py`, `test_quixicore_moe_sum_add.py`, `test_f32_overrides.py` | ~600 | with their kernels |
| bring-up patch (platform `rtx6000`, record, DeepGEMM cmake guard) | ~75 | Phase 0 |

ALREADY UPSTREAM (only an env or a capability gate is needed): custom
all-reduce over PCIe and its size cap (`VLLM_CUSTOM_AR_ALLOW_PCIE`,
`VLLM_CUSTOM_AR_PCIE_MAX_BYTES`); the FP32 router GEMV (`dsv4_router_gemm`,
gated to SM80 and 288 experts: widen after parity); KDA o_norm through
Triton, g_a merged into in_proj, paired f_b / g_b.

DROP: the small-k sampler (`topk_sample.cuh` 413 lines + 200 in
`sampler.cu`: a tie-handling argument, not a measured throughput item;
revisit only if the Phase 0 trace ranks sampling); the mHC last-block
variant (neutral); H16 swapAB sparse prefill and tensor-core mHC prefill
(0.3-0.5 % TTFT; the latter failed its quality gate); `glm_moe_stable_align.cuh`;
the publish/consume-by-tensor-identity routing wiring (rewrite the ~60
lines); `weight_recipe.py`'s recipe-directory mechanism; every
`slimserve/*_journal.py`, `*_diagnostic.py`, `canonical_moe.py`,
`glm53_ordering.py`, `reduction_receipts.py`, `rmsnorm_*.py`,
`prompt_score_*.py`, `index_journal.py`; the campaign harnesses and the
notebook history.

Exported copies of the KEEP set: `~/.local/scratch/slimserve-glm53/salvage/`
(`csrc/`, `python/`, `patches/`). The local branch and the fetched PR refs
are deleted; the PR heads remain on GitHub as `refs/pull/{24..28}/head`.

## 2. Where the tree stands (upstream/main 2b355117e, 2026-09-11)

GLM-5.3-Flash on A100 (`glm53f-nvfp4-4` TP4, `glm53f-nvfp4-8` TP4 x DP2 with
replicated MoE) is an active campaign on the 8x A100 box; its handoff is the
last section of `HANDOFF.md`. Present and validated there:

- `vllm/model_executor/models/glm5_next.py` (mHC sites, EAGLE-3 aux taps for
  DFlash2, torch.compile with `kda_attention` as an opaque op, MTP block for
  the checkpoint head), `glm5_next_indexer.py` (pooled indexer, compact
  cache, TP row sharding), `kimi_gdn_linear_attn.py` (KDA with the merged
  in-projection, paired gates, Triton o_norm), `quixicore_mla_sparse` backend
  with `vllm/quixicore/sparse_mla_tc.py` (tensor-core NoPE sparse decode).
- QuixiCore serving kernels under `csrc/quixicore/serving/`: `mla_kernels.cuh`
  (partitioned sparse NoPE decode, P128 with a 512 MB scratch gate),
  `paged_attn_v2_kernels.cuh` (reducers incl. `paged_attention_reduce_channels`),
  `mhc_ampere.cuh` / `mhc_channel_owned_ampere.cuh` / `mhc_allreduce_ampere.cuh`,
  `dsv4_router_ampere.cuh` / `dsv4_projection_ampere.cuh` (bf16-in fp32-out
  GEMVs), `indexer_logits_mma.cuh`, `glm53f_warp_histogram.cuh`, the DSV4
  Marlin-lesson quant kernels under `quant/`. All compiled for
  `QUIXICORE_ARCHS = 8.0;8.6;8.9;9.0;10.0;11.0;12.0` intersected with
  `TORCH_CUDA_ARCH_LIST`, so `12.0f` builds without CMake changes
  (`tools/build_deepgemm_C.py` is absent and the DeepGEMM target must be
  guarded, as 6b062ad3f did).
- Registry: profile ids are `glm53f-nvfp4-4` / `glm53f-nvfp4-8` (renamed
  2026-09-06; plain `glm53` is the 743B model). Source `glm53f-nvfp4` names
  the RedHatAI checkpoint and the `incoai/GLM-5.3-Flash-DFlash2` drafter
  (cc-by-nc-nd-4.0, block 8, taps layers 5/14/24/33/42, served through the V2
  DFlash2 speculator). A100 record engine: TP4, EP off, Marlin experts,
  `QUIXICORE_MLA_SPARSE` + `sparse_mla_force_mqa`, block 64, `kv_cache_dtype
  auto` (bf16) with a per-layer fp8 opt-in for the sparse MLA layers,
  FULL_DECODE_ONLY graphs, prefix caching, host + NVMe KV tier, thinking
  budget 2000, and the `additional_config` switches
  `glm5_next_{compact_indexer_cache, shared_indexer_scratch,
  singleton_marlin_alignment, mhc_projection_overlap, singleton_pool_update,
  sparse_tc_decode, indexer_row_shard}`.
- A100 numbers (exact-token 1000/300, warmed, median of three): TP4 87.4 /
  368.4 / 502.9 (c1/c8/c16, corrected router, 2026-09-09); TP8 DFlash2 k=3
  record 153.6 / 495.0 / 687.3 / 879.5 (c32) / 1074.5 (c64) vs its non-spec
  109.9 / 494.6 / 676.0.

Missing or gated for sm_120 (each is a Phase 0/1 item):

- No `rtx6000` platform in `slimserve/hardware.py:_classify` or
  `profiles.json`, no `min_gpus.rtx6000`, no variant record.
- SM80-only predicates that must be qualified, not blindly widened:
  `glm5_next_indexer.py` (row sharding "requires SM80"; compact cache
  "qualified only on SM80"), `glm5_next.py:366` (norm fused inside the mHC
  transition only on the measured SM80 path), `vllm/quixicore/sparse_mla_tc.py`
  (SM80 tensor-core sparse decode), the fp8 split cap in the sparse TC decode
  (SPLIT=128 exceeds sm80 smem; sm_120 has LESS: 99 KB opt-in vs 163 KB).
- Every ~90 per-step all-reduce runs on the pynccl ring unless the custom AR
  escape hatch is set (world 4 without NVLink disables it by default).
- Standing policy (CLAUDE.md, 2026-09-10): V2 model runner only (V1 is
  deprecated); FP8 is the only permitted KV quantization, TurboQuant is
  prohibited; prefix caching, tool calling and thinking are always on; never
  greedy sampling in a profile.

## 3. Hardware and physics (measured 2026-09-04 unless marked)

RTX PRO 6000 Blackwell Workstation: 188 SMs, 128 MiB L2, 100 KB smem per SM
(99 KB opt-in per block), 96 GB GDDR7 at 1.79 TB/s nominal (~1.6 TB/s
practical streaming, lm_head measured 1.58-1.63), 3090 MHz boost, 600 W.
Topology: GPU0-GPU1 and GPU2-GPU3 under one host bridge each (PHB), cross
pairs through the root complex (NODE); P2P reads work on every pair (52 GB/s,
8 KB copy 9.5 us). Host: EPYC 9334 32c/64t, one NUMA node, 188 GB RAM,
earlyoom active. CUDA 12.8 / 13.0 / 13.2 toolkits; the driver is 13.0, so
CUDA 13.3 images run handicapped (NCCL cuMem/P2P and CUDA IPC fail).

GLM-5.3-Flash: 45 layers (3 dense FFN 12288, 42 MoE: 288 routed experts
top-8 at intermediate 2048 = 25.2M params each, plus one shared expert), 34
KDA linear-attention layers (64 x 128 heads, 2 MiB state per layer per
sequence), 11 DSA sparse-MLA layers (64 heads, 256 nope, no RoPE,
kv_lora_rank 512, indexer 32 x 128 with 4x key pooling, top-2048), mHC
hyper-connections (4 residual streams), vocab 154,880, one MTP layer, 1M
positions.

Bytes per token per GPU at TP4 and the resulting no-speculation c1 ceiling:

| backbone format | experts/GPU | backbone/GPU | per GPU | floor/step | c1 ceiling |
|---|---|---|---|---|---|
| RedHatAI as shipped (BF16 backbone) | 1.19 GB | 4.59 GB | 5.78 GB | 3.2 ms | ~310 tok/s |
| + native FP8 dense/shared/DSA (swap-set) | 1.19 | 3.87 | 5.05 | 2.8 | ~355 |
| + FP8 KDA q/k/v/o (self-quantized) | 1.19 | 2.72 | 3.91 | 2.2 | ~460 |
| + kv_b and lm_head FP8 | 1.19 | 2.52 | 3.71 | 2.1 | ~480 |
| NVFP4 everything | 1.19 | ~1.3 | ~2.5 | 1.4 | ~720 |

At c8 the routed experts dominate: up to 64 distinct experts per layer x 3.4
MiB per rank x 42 layers = 9.1 GB per rank = 5.1 ms per step at nominal
bandwidth; Marlin W4A16 was measured AT that floor (5.17 ms). So the c8
no-speculation ceiling with an FP8 backbone is about 7.3 ms per step, ~1100
tok/s, and only overlap, launch count, mHC and the M=8 GEMM efficiency move
it; at c1 the experts are 1.1 ms against a 0.63 ms floor (63 %) and a
decode-shaped kernel is worth ~0.4 ms.

Communication: ~90 all-reduces per step (2 per layer). NCCL ring LL over P2P
is 10.3-11.2 us at c1 (0.93-1.0 ms/step); vLLM's custom AR over P2P is 5.1
us (0.46 ms/step); a one-shot PCIe-IPC kernel might reach 3-4 us. Prefill
chunks above the custom-AR size cap fall back to NCCL (37 % of a 7K-token
prefill step until the cap was raised).

Where the last measured fast-state c1 step went (5.53 ms at 162.8 tok/s):
FP8 decode GEMMs 1.87 ms (176 launches), Marlin experts ~1.1, mHC 0.78 (91
sites x 8.5 us for ~1.5 MB each: latency, not bytes), custom AR 0.46, bf16
GEMMs 0.24, pooled indexer 0.20, KDA recurrent+conv ~0.2, sparse MLA 0.13,
launch tail ~0.5, in-step idle 0.18. Against a 2.2 ms byte floor the
engineering target for c1 is 3.2-3.5 ms (285-310 tok/s) before speculation.

## 4. Controls, bars and targets

All numbers are exact-token `benchmarks/benchmark_dsv4_exact.py` at 1000 in
/ 300 out, temperature 1.0 / top-p 0.95 / top-k 20, seed 42, warmed, aggregate
output tok/s at c1 / c8 / c16 unless stated. Protocol matters: the published
B12X table is 30 s context-zero cells with other serving active, which is not
our shape.

| line | c1 | c8 | c16 | source |
|---|---:|---:|---:|---|
| B12X R24 published, no spec (their protocol) | 169.9 | 737.8 | - | rtx6kpro, 2026-09-04 |
| B12X R24 published, MTP-3 | 247.8 | 903.2 | - | 2.50 accepted/step |
| B12X R24 on this box, handicapped (NCCL over SHM), our protocol | 134.0 | 483.7 | 598.6 | notebook 2026-09-04 |
| B12X R28.1 on this box, cold-prefix E2E protocol | 132.6 | 466.7 | 589.3 | Codex review doc |
| A100 TP4 `glm53f-nvfp4-4` | 87.4 | 368.4 | 502.9 | 2026-09-09 |
| A100 TP4xDP2 `glm53f-nvfp4-8`, DFlash2 k=3 | 153.6 | 495.0 | 687.3 | 2026-09-10 |
| Codex best, no spec (fast-state boot) | 162.8 | 534.5 | 691.0 | 2026-09-08 |
| yhfgyyf vLLM fork, TP4, FP8 KV, MTP-5, 8K in / 512 out | 158 | - | - | prefill 9.9K tok/s; 2026-09-07 |
| SGLang cookbook (FP8, 4x PRO 6000) | 63.7 | - | - | rtx6kpro |

Gates for this campaign (in order of strictness):

1. Floor (mandatory): beat the local B12X control in every cell on our
   protocol, including a fresh control run in Phase 0 on the newest image
   that boots on this driver.
2. Target (no speculation): c1 >= 250, c8 >= 700, c16 >= 850, cold 32K
   TTFT <= 2.0 s, cold 128K TTFT <= 8 s. These are 65-75 % of the physics
   ceilings in section 3 with the FP8 backbone.
3. Target (with the chosen speculator, on the Foundry-like structured
   workload and on prose): c1 >= 320 where acceptance supports it; c8 no
   worse than no-spec (the schedule turns speculation off above the batch
   size where it loses, as the A100 record does).
4. Stretch: the published B12X MTP-3 numbers on their protocol, measured
   with the same protocol on our stack.
5. Sanity: TP2 vs TP4 on the same build >= 1.5x at c8 (CLAUDE.md gate; TP2
   is a measurement, not a record); c8/c1 ratio >= 3.5 at NVFP4.

## 5. Research digest (borrow / avoid), refreshed 2026-09-11

Borrow, with the exact precedent to read before implementing:

- vLLM #53906 (merged 09-03): native GLM-5.3-Flash incl. the MTP draft model;
  #55214 the missing `__init__.py`. Upstream SlimServe already has the MTP
  block and the DFlash2 path; use these only to cross-check.
- vLLM issue #53963 (2026-08-26, open): GLM's rope-free sparse MLA has no
  SM120 path in stock vLLM (fp8_ds_mla asserts pe_dim 64; BF16 lane excluded;
  kernel shape guard). SlimServe's QuixiCore NoPE kernel is the answer; the
  community fork `yhfgyyf/vllm-deepseek-v4-sm89` (Apache-2.0, updated
  2026-09-07) ships a `FLASHINFER_MLA_SPARSE_SM120` backend with FP8 KV for
  GLM and reports 158-199 tok/s decode with MTP-5 and 9.9K tok/s prefill at
  TP4: the closest external comparison on this exact card, read its kernel
  choices for the sparse path and its KV layout.
- vLLM #55277 (open) + flashinfer #4802 (merged 09-03, `GLM53_NOPE`, decode
  14.9 -> 11.3 us) + #4947: the SM120 sparse NoPE MLA path and the
  `fp8_ds_mla` row layout for pe_dim 0. A/B partner for our kernel; also the
  reference for FP8 KV on the NoPE row.
- vLLM #53576 (merged 09-01, tested on 4x RTX 6000 Pro TP4): FlashInfer
  PCIe-IPC all-reduce (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`, FlashInfer
  > 0.6.17 nightly wheels). Ceiling ~0.25 ms/step over the custom AR we
  already run at 5.1 us; Phase 2, after the cheaper levers. Never build
  FlashInfer from source on this box.
- KDA: flashinfer #4709 (merged 09-03) `fused_kda_decode` (conv + SiLU +
  recurrent update + gated RMSNorm in one SM100 launch, 1.5-5.9x), vLLM #55364
  (open) FlashInfer KDA decode 2.47x at bs1, #54697 (merged) overlapped low-M
  KDA projections, SGLang #37744 qkvbfg 6 -> 2 GEMMs per KDA layer. The
  fused-decode structure is the Phase 2 KDA item; keep the FP32 forward
  substitution (FlashKDA R21 emitted non-finite values without it).
- vLLM #40082 (merged 2026-05-20): FlashInfer b12x fused MoE
  (`flashinfer_cutedsl_sm12x`) and FP4 GEMM (`flashinfer-b12x`) for
  SM120/121, CuTe DSL warp-level MMA tuned for small-M decode; needs
  nvidia-cutlass-dsl 4.4.2 exactly. This is the W4A4 expert path B12X uses.
  Our vLLM base predates it; port only the kernel idea (weight-streaming
  small-M FP4 with adaptive tiles), not the backend, and gate on the U+FFFD
  and exact-token checks because CUTLASS grouped FP4 on SM120 produced
  garbage in stock vLLM (cutlass #3096, vLLM #54150; a patched native path ran
  39 tok/s vs Marlin 46-49 there).
- vLLM #55170 (open): CUTLASS W4A4 over W4A16 for NVFP4 linears on SM120,
  +23.8 % on RTX PRO 6000 Max-Q; flashinfer #4718 sm12x NVFP4 MoE retune;
  #4013 drops tiles over 101 KB smem. Prefill-side references.
- vLLM #54951 (draft, TP-sharded indexer prefill): already in SlimServe
  (SM80-gated). vLLM #54110 (persistent top-k fallback for < 128 KiB smem,
  tested with GLM-5.3-Flash MTP): read for the sm_120 smem budget.
- local-inference-lab/vllm #576: `cp.async.bulk.prefetch.L2` of next weights
  during reduction windows (+7 % c1 at bf16). Codex's 8 MiB variant regressed
  c8/c16; only a next-layer variant with a measured overlap window qualifies.
- llama.cpp #27970 (gather-then-dense FA over indexer-selected KV, 2.2x at 1M)
  and #27917 (MTP acceptance per position 0.92 / 0.81 / 0.60 / 0.44 / 0.35:
  depth 3 is the sweet spot). exllamav3 #330: BF16 I/O on the decode GEMM.
- B12X recipe (rtx6kpro `models/glm5.md`, qualified 2026-09-04): TP4 DCP1,
  ModelOpt NVFP4 W4A4 experts, FP8 KV (packed NVFP4 KV selectable),
  FULL_AND_PIECEWISE graphs, 4,096 target tokens per step, compute-share
  fairness 0.4, MTP depth 3 or DFlash2 depth 7 (ModelOpt MXFP8 drafter),
  two MoE paths by concurrency.
- barrydeen/glm53-flash-dgx-spark (sm_121, 2x GB10): patch stack for NVFP4
  at 262K with fp8 KV and DFlash2; the closest open recipe for the DFlash2 +
  fp8 KV combination on the 12.x family.
- In-repo precedents to read before each phase: `csrc/quixicore/
  a100_glm52_design.md` (Marlin lessons: layout-only repack, in-register scale
  decode, cp.async staging, tensor cores for M >= 2, fusion of decode nodes),
  `dsv4_ampere_design.md` and `perf/dsv4_a100_kernel_history.md` (what was
  measured slower: output-owned hidden state, channel-residual mHC, fused
  shared-expert publication, cooperative peer consumption, indexer head
  sharding, top-512 radix, projection bundling), `perf/metal_m1ultra_*`
  (the campaign shape the operator wants: research, ranked items, one entry
  per experiment), the A100 GLM-5.3 entries of 2026-09-07..10 in
  `perf/optimization_status.md` (KDA norm/projection fusion, SIMT mHC, FP32
  router, compact indexer, sparse-TC decode, prefill row map).

Avoid (measured or documented dead ends):

- Re-writing cuBLAS-class dense BF16 GEMMs for their own sake: SGLang #37899
  measured +4 % op-level and ~0 % end to end. The lever is bytes (FP8) and
  launch count; the M<=16 decode GEMM was worth 1.5 %, not more.
- Fused all-reduce + mHC transition: rejected on A100 (85.5/421 vs 84/413);
  the mHC math is replicated on every rank, so the win is ownership, not
  merging two kernels.
- EP experts: 4-6 % slower than TP experts at every A100 config.
- ModelOpt NVFP4 checkpoints (LibertAIDAI, dealignai): U+FFFD emission on 4x
  PRO 6000 (vLLM #54150); ModelOpt weight-only MoE input_scale folded to zero
  (#54189). RedHatAI compressed-tensors was clean. The checkpoint is fixed.
- FP8 KV on SM120 through SGLang's path: garbled output or stops after one
  token (rtx6kpro notes). Our fp8 NoPE-row kernel is a different path but
  gets the same gate.
- Torch's P2P assumption is wrong for some PCIe sm_120 copies (exllamav3
  #333). Validate every peer copy path explicitly.
- Reduction order changes acceptance (SGLang #37746). Measure acceptance
  before and after any all-reduce change.
- Hybrid page alignment (#54458) inflates blocks and collapses prefix-cache
  hits; check the block geometry when KV groups change (speculation adds
  groups).
- flashinfer #4827: sm12x MoE workspace use-after-free under CUDA graphs on
  GLM-5.3-Flash TP2. Any FlashInfer MoE path gets a graph-replay soak.
- Measurement anti-patterns: `--ignore-eos` TPS, synthetic repeated-token
  prompts, interval logs as benchmarks, MoE microbenchmarks with random
  `topk_ids`, sums as fingerprints, TPS without acceptance for spec runs,
  env-gated A/B arms under torch.compile (the compile cache key ignores
  private env; A/B via distinct code or `graph_factors`).

## 6. The regimen

This is the loop, in the order the operator described it. Each roadmap item
in section 9 goes through it once.

1. Research (time box: 1 hour per item, 1 day for the campaign's initial
   pass). Name the precedent: a PR, a kernel in a reference tree
   (`~/llama.cpp`, `~/ds4`, `~/QuixiCore/*`, vLLM Marlin, FlashInfer), or a
   SlimServe notebook entry from another platform. State what transfers to
   this hardware and what does not (smem 99 KB, 188 SMs, PCIe topology).
2. Rank by expected end-to-end gain, computed from the current step
   attribution (kernel ms per step per class, launches per step), not from
   isolated kernel speedups. Re-rank after every phase from a fresh profiler
   pair.
3. Design note (10 lines, in the notebook entry's Hypothesis): what changes,
   what the trace should show afterwards, the expected ms/step, the
   correctness contract (bit-exact, summation order, or a gate).
4. Implement in `csrc/quixicore/` (kernels) and the model/layer files; keep
   the old path selectable through one kill switch that is also a compile
   cache factor (`graph_factors`), and remove the switch once the change is
   retained and rebuilt.
5. Correctness: parity test in `tests/kernels/` against an fp32 reference on
   the real shapes (existing pattern:
   `tests/kernels/test_quixicore_sparse_mla_bf16.py`); microbench at the real
   serving shapes with weights rotated past L2 and launches inside a captured
   graph (`gemm16_bench.py` pattern); report GB/s against 1.79 TB/s.
6. End to end, through the profile only: one profiler pair for launch-count
   changes; one exact-token boot per arm (two when the expected delta is under
   3 %); the state label on every boot; the quality gate on the candidate
   boot; the canaries. Compare like-state boots only.
7. Notebook entry (Status / Scope / Baseline / Hypothesis / Change /
   Correctness / Results / Decision / Raw artifacts) and a commit, retained
   or rejected. Rejections are one paragraph.
8. Next item. Do not stop the loop at a commit; stop at a phase gate (section
   9) for the operator's review, or when a decision listed in section 11 is
   needed.

Rules that come from the previous campaign's failure:

- No validation campaign without a retained change to validate. One gate set
  per retained change (section 7). Reproducibility, determinism, sampler
  semantics and scoring-tool audits are out of scope unless a gate fails,
  and then the fix is time-boxed to half a day and the gate is re-run once.
- No diagnostic tooling in the repo. Instruments live in
  `~/.local/scratch/slimserve-glm53/`; the repo gets kernels, their tests,
  profile records and notebook entries. Failed experiments are removed from
  the serving code, not left behind switches.
- Sweeps are not the method. A sweep is at most one hour of microbench over
  at most three parameters after the design has been chosen from research
  (tile rows x stages, partition size, a launch table). Never a serving sweep
  of configurations.
- Every item has a time box (section 9) and a stop rule: if the profiler
  pair shows under 1 % and the exact-token boot is neutral, park it with a
  one-paragraph entry and move on. Parked items are revisited only with a new
  hypothesis.
- A day with no change to a kernel or the serving path is a red flag. Report
  it in the notebook with the reason and the decision that unblocks it.
- The B12X control is re-run once per phase at most, and only on the same
  protocol as our number of the day.
- Speculation numbers always carry accepted tokens per step and the step
  rate (tok/s divided by mean accepted length); acceptance is measured on the
  operator's real workload shape (structured JSON, thinking on, 8-wide) as
  well as on the prose harness.

## 7. Measurement protocol (exact commands)

Serve (the only recorded path):

```bash
# operator env (kept in ~/.local/scratch/slimserve-glm53/serve.sh, not in the repo)
export SLIMSERVE_CACHE=/raid/weights
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export VLLM_CACHE_ROOT=/home/tiny/.local/scratch/slimserve-glm53/vllm-cache
export TMPDIR=/home/tiny/.local/scratch/slimserve-glm53/tmp
.venv/bin/python -m slimserve.cli glm53f-nvfp4-4 --dry-run          # inspect the rtx6000 record
systemd-run --user --scope -p MemoryMax=150G --quiet -- \
  .venv/bin/python -m slimserve.cli glm53f-nvfp4-4 --serve --host 127.0.0.1 --port 8000 -y [--no-spec] [--torch-profile-dir DIR]
```

Throughput (repo canonical shapes; c1, c8, c16 at 1000/300, plus 1000/2000
at c1 and c8 once per phase; upstream's harness flags, `--help` first):

```bash
.venv/bin/python benchmarks/benchmark_dsv4_exact.py \
  --model /raid/weights/GLM-5.3-Flash-NVFP4 \
  --source /home/tiny/.local/scratch/slimserve-glm53/prompt-source.txt \
  --url http://127.0.0.1:8000/v1/completions \
  --concurrency 8 --input-tokens 1000 --output-tokens 300 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --seed 42
```

Three warmed repeats per cell, median reported, spread recorded (upstream's
A100 protocol); the second and third repeats are APC-hot. Raw JSON and the
completions go under `perf/results/YYYY-MM-DD/<run-id>/` with `run.txt`
(commit, dirty count, `nvidia-smi` clocks/power per shape).

Boot state label (mandatory on every recorded boot): one profiled c1 round
after the benches (`prof_run.py` + `gap_locate.py`); in-step idle under ~0.2
ms with custom AR is the fast state. Arms are compared like-state; a
slow-state boot is reported, not averaged, and a slow boot on a quiet host is
the next thing to profile. Log load average and top CPU processes at every
pass (`ab.sh` does this).

Profiler pair (decision instrument for anything expected under ~3 %):
`prof_pair.sh <label> "<envA>" "<envB>"`, one c1 capture per arm, full decode
steps only, compare GPU span, kernel busy, in-graph gaps and launches per
step (`step_attrib.py`). Per-class tables go in the notebook.

Correctness gates before a change is retained:

- `gate.py --out perf/results/<date>/<run>-gate.json`: mean prompt logprob on
  8 fixed 512+32-token slices (band on this stack: -2.407..-2.478 nats,
  within-boot jitter 0.02-0.03, so deltas under 0.03 are noise) and needle
  margins at 1K and 7K (pass/fail).
- Text (+reasoning), image and tool canaries through the chat endpoint; 0
  U+FFFD across the benchmark completions; repeated-window degeneration check.
- Kernel parity tests in `tests/kernels/` for every kernel change; sanitizer
  (`compute-sanitizer --tool memcheck` and `racecheck`) on new kernels once.
- Speculative runs: accepted per step from the server counters and a
  degeneration guard on every run.
- Once per phase: the deep-context leg (needle recall to 131K through the
  profile; the A100 harness `benchmarks/benchmark_wildchat_deepcontext.py`),
  and when the KV tier is enabled, forced-eviction acceptance with
  `VLLM_KV_TIER_VERIFY=1` (0/N mismatches is the pass, not marker recall).

Build (Codex's working recipes; 73 min full, ~90 s incremental):

```bash
# full editable rebuild (no serving or benches while it runs)
CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=12.0f MAX_JOBS=8 NVCC_THREADS=2 CMAKE_BUILD_TYPE=Release \
VLLM_CUTLASS_SRC_DIR=/home/tiny/.local/scratch/slimserve-glm53/cutlass TMPDIR=/home/tiny/.local/scratch/slimserve-glm53/tmp \
systemd-run --user --scope -p MemoryMax=120G --quiet -- \
  ~/.local/bin/uv pip install --python .venv/bin/python -e . --no-build-isolation -v
# incremental _quixicore_C in a persistent cmake dir: ~/.local/scratch/slimserve-glm53/native_incremental.sh [_quixicore_C]
# then: python -c "import vllm._C_stable_libtorch, vllm._quixicore_C" and the kernel tests on GPU 0
```

## 8. Rubric

Retain a change only if all of these hold:

1. Correctness: parity test passes; gate mean within the band or better;
   canaries pass; exact-token `exact:true` in every cell; no U+FFFD; for spec
   runs, acceptance recorded and no degeneration.
2. Throughput: c1 and c8 on the canonical shape within noise or better
   between like-state boots, c16 not regressed beyond the spread; the
   profiler pair shows the mechanism the hypothesis named (fewer launches,
   shorter class, smaller gaps). A change that is faster for a reason other
   than its hypothesis is a finding, not a retained change, until the reason
   is understood.
3. Prefill: cold 32K and 128K TTFT not regressed (once per phase, and for any
   change that touches prefill).
4. Code: one kill switch or none; no second copy of a serving path; no
   diagnostic-only branches; the switch is a compile-cache factor; a test
   exists for the shape gate (which M, N, K, dtype take the new path).
5. Notebook: the entry exists with raw artifact paths before the commit.

Reject (and record) when any of the above fails after the time box. Park
when neutral. Never retain on a best-of-boots number.

PR readiness (Phase 6): every retained change has its entry; the record row
in `perf/baseline_status.md` is measured on the PR's tree with three repeats;
the `rtx6000` record's notes state what each setting is and why; the
registry tests pass (`tests/slimserve/`); kernel tests pass on GPU 0; no
scratch paths, host names or deploy config in the diff; under ~100 files per
PR (split by layer if larger: kernels, serving integration, record +
notebook); commits authored as Auroter with no trailers.

## 9. Roadmap

Time boxes are working days on this machine; the loop in section 6 applies
to every item. Expected gains are against the last attribution (section 3)
and are re-derived from the Phase 0 trace before Phase 1 starts.

### Phase 0: branch, build, platform, baseline (1 day)

1. Native build of this tree for `12.0f` (full rebuild; the `.so` files in
   `vllm/` are from the deleted branch). Re-apply the DeepGEMM cmake guard.
   Smoke `import vllm._C_stable_libtorch, vllm._quixicore_C`; run the
   existing GLM kernel tests on GPU 0 (`tests/kernels/test_quixicore_sparse_mla_bf16.py`,
   `tests/glm5_next/`), noting which are SM80-gated.
2. Platform `rtx6000` (`hardware.py:_classify` on "rtx pro 6000",
   `profiles.json` platform with compute capability [12, 0] and the notes
   from 6b062ad3f, `min_gpus.rtx6000: 4` on the NVFP4 quant) and the
   `glm53f-nvfp4-4` `rtx6000` variant record: TP4, EP off, Marlin experts,
   `QUIXICORE_MLA_SPARSE` + `sparse_mla_force_mqa`, block 64, bf16 KV,
   FULL_DECODE_ONLY capture 64, max_num_seqs 16, prefix caching, thinking
   budget 2000, parsers glm47, vision on, speculation off for the baseline,
   no KV tier yet, every `glm5_next_*` additional-config switch off until
   qualified. Registry tests pass (`test_a_profile_is_one_config_per_platform`).
3. Environment audit, recorded in the entry: `NCCL_P2P_DISABLE` (Codex found
   `~/.config/fish/conf.d/*.fish` exporting it box-wide, which forces NCCL
   through host SHM), CPU governor, GPU clocks and power limits, `nvidia-smi
   topo -p2p r`, free host RAM, earlyoom threshold.
4. Baseline through the profile: three warmed repeats of c1/c8/c16 at
   1000/300, 1000/2000 at c1 and c8, cold 32K and 128K TTFT, state label,
   gate, canaries, one profiler pair capture (c1 and c8) for the attribution
   table. Record in `perf/baseline_status.md`.
5. Fresh B12X control on the same protocol with the newest image that boots
   on this driver (R24 is pulled; check rtx6kpro for a newer tag). If the
   operator upgrades the driver (section 11), re-run unhandicapped.

Exit: profile healthy, baseline row written with the attribution, every
SM80-gated feature listed with its sm_120 status (works / needs smem
retune / not applicable).

### Phase 1: bytes and the cheap fixed-overhead wins (2 days)

Re-land from the previous campaign, one commit and one entry each, each
re-measured (expected gains are Codex's like-state deltas):

1. F32 sidecar loader hook (correctness; neutral).
2. Custom all-reduce over PCIe on the record env (+11 % c1) and the AR size
   cap for prefill chunks (-14 % c8 prefill); both knobs exist upstream
   (`VLLM_CUSTOM_AR_ALLOW_PCIE`, `VLLM_CUSTOM_AR_PCIE_MAX_BYTES`), so this is
   record env plus a measurement.
3. bf16 M<=16 decode GEMM custom op (+1.5 % c1).
4. FP8 swap-set (dense/shared/DSA; +5.5 % c1) then FP8 KDA projections
   (+10.6 % c1, NLL gate mandatory; ZAI kept these BF16).
5. Sparse MLA partition rule + channel reducer (+3.7 % c1).
6. Fused routing + alignment, combine, Marlin workspace (-160 launches).
7. Qualify upstream's SM80-gated wins on sm_120 by measurement, widening each
   predicate only after its parity test and smem check: FP32 router, SIMT mHC
   with fused norm, compact indexer cache, sparse tensor-core decode, TP
   indexer row sharding.
8. FP8 (e4m3) main KV for the 11 sparse MLA layers (upstream's per-layer
   opt-in) on the NoPE row: parity, the U+FFFD and needle gates, then the
   deep-context leg. Small decode gain at 1K context; the win is capacity and
   long-context prefill, and it is the only permitted KV quantization.

Exit: like-state c1 >= 165, c8 >= 540, c16 >= 700 (the previous campaign's
best, now on upstream's tree with its A100 wins on top); FP8 KV decision
recorded; fresh attribution.

### Phase 2: fixed overhead, the decode-step levers (3-4 days)

Ranked by ms per step at c1 from the section 3 attribution; re-rank from the
Phase 1 trace.

1. mHC transition (0.78 ms at c1, 1.13 ms at c8 where T>1 sites run three
   launches): one launch per site at every T with the partials and the
   Sinkhorn tail in one block per token where co-residency allows, or the
   DSV4 channel-ownership form fused into the custom all-reduce (each rank
   transitions 4096/TP channels; `mhc_channel_owned_ampere.cuh` and
   `mhc_allreduce_ampere.cuh` are the precedents, `VLLM_DSV4_TP_OWNERSHIP`).
   Target 8.5 -> 3-4 us per site: -0.4 to -0.5 ms at c1, -0.7 ms at c8.
   Time box 1.5 days.
2. Fused KDA decode: in_proj (FP8) -> short conv -> gate -> delta-rule state
   update -> gated RMSNorm -> o_proj as two or three launches per layer instead
   of ~11 (flashinfer #4709 `fused_kda_decode` is the structure reference,
   FP32 substitution kept). Whole-head fusion regressed before; the fusion
   boundary must be chosen from the trace (projections stay separate if they
   are at roofline). Target -0.3 ms at c1. Time box 1.5 days.
3. Launch tail: MoE glue after routing (Marlin workspace fill, act, sum),
   norm/add fusions, copies around the mHC wrappers; each removed launch is
   worth its time plus ~0.5 us of dependency latency inside the graph. Target
   -0.3 ms. Time box 1 day.
4. M=8/16 GEMM efficiency on the FP8 backbone (c8 lever: the fp8 decode GEMMs
   at M=8 and the shared-expert shapes under Marlin contention). Target -0.3
   ms at c8. Time box 0.5 day.
5. PCIe-IPC one-shot all-reduce (FlashInfer #53576 port, nightly wheels only)
   only if 1-4 leave the AR as the largest fixed cost; ceiling -0.25 ms.

Exit: c1 >= 220 no-spec, c8 >= 620, launches per step and in-graph gaps
reported before/after; TP2 vs TP4 sanity measured once.

### Phase 3: speculation (1-2 days)

1. DFlash2 on the V2 runner (upstream's registered speculator; block 8; the
   compact indexer cache caps k at 5): measure k=3 and k=5 with the A100
   per-batch schedule pattern (`num_speculative_tokens_per_batch_size`) on the
   prose harness AND on the operator's structured 8-wide workload; acceptance
   and step rate recorded. Check hybrid page geometry (#54458) after the
   drafter adds KV groups.
2. Checkpoint MTP head (layer 45, in-tree block) as the A/B: k=3, same
   workloads. Codex measured +17..46 % greedy structured, -15..30 % prose at
   temp 1.0, -20 % at Foundry c8; upstream measured +6.5 % c1 on A100 V1.
3. Decision per record: the speculator and schedule that wins the operator's
   workload without losing c8; license of the DFlash2 drafter (cc-by-nc-nd)
   is flagged for the operator.

Exit: c1 with speculation >= 300 on the structured workload, c8 >= no-spec,
acceptance recorded; deep-context leg re-run once because group geometry
changed.

### Phase 4: sm_120-specific kernels (3-4 days)

1. Decode-shaped NVFP4 expert kernel for M<=16 per expert: weight-streaming
   over the union of active experts, in-register group-16 scale decode, fused
   SwiGLU and top-k weighting, cp.async ring retuned for 99 KB smem and 188
   SMs (`a100_glm52_design.md` and the fused Q4_K decode pair are the
   precedents; flashinfer sm12x retune #4718 and the b12x CuTe kernel are the
   external references). Target >= 85 % of 1.79 TB/s at c1 (Marlin is at 63
   %): -0.4 ms at c1, neutral at c8/c16 where Marlin is already at the byte
   floor. Two paths by concurrency if the data says so. Gate: exact-token,
   U+FFFD, graph-replay soak. Time box 2 days.
2. Head-batched sparse MLA decode: read each selected latent row once for
   the 16 heads per rank instead of once per head (at c8/c16 the kernel is
   L2-bound on that re-read: 22-44 us per layer for 1-2 MB of distinct KV);
   pooled indexer `_pooled_logits_kernel` at 17 us per layer for 1000
   tokens. Target -0.15 ms at c8, more at long context. Time box 1 day.
3. lm_head at 89 % of streaming and the remaining bf16 shapes: FP8 lm_head
   and kv_b (section 3 table, ~0.1 ms) behind the NLL gate. Time box 0.5 day.

Exit: c1 >= 250 no-spec, c8 >= 700, every kernel with parity test,
microbench GB/s and an e2e entry.

### Phase 5: prefill (2 days)

1. Attribution of a cold 32K and a cold 128K prefill (indexer, sparse
   attention, Marlin M>=64, all-reduce share, KDA chunked scan).
2. TP-sharded indexer prefill and the H16 sparse prefill (qualified from the
   previous campaign's evidence), CUTLASS sm120 block-scaled FP8 GEMMs for
   the FP8 backbone at prefill, W4A4 tensor-core experts for M>=64 if the
   gated FP4 path is clean.
3. Targets: cold 32K TTFT <= 2.0 s, cold 128K <= 8 s, prefill >= 12K tok/s
   at 32K (yhfgyyf 9.9K, B12X 14.9K).

### Phase 6: retention and PR (1 day)

1. Record row with three repeats on the final tree; `rtx6000` record notes
   rewritten to state each setting's evidence; kill switches removed for
   retained changes; failed paths deleted.
2. Host + NVMe KV tier on this box (188 GB RAM: 32 GiB host per rank + a disk
   tier under `/raid`), qualified with forced eviction and
   `VLLM_KV_TIER_VERIFY=1` (operator decision on sizes, section 11).
3. PR(s) against QuixiAI/SlimServe main, under 100 files each; port retained
   kernels to QuixiCore-CUDA (`bf16/fp8_decode_gemm.cuh`, routing/combine,
   mHC, sparse reducer, the NVFP4 decode kernel) with their tests.

## 10. Standing machine constraints

- One GPU workload at a time; no native builds while a server or bench
  runs; attribute GPU processes by PID lineage. The four GPUs are idle today
  (Foundry stopped).
- Venvs under `~/venvs` (never on /raid); scratch, logs, build trees and
  traces under `~/.local/scratch/slimserve-glm53/` (never /tmp); model weights
  under `/raid/weights` (`SLIMSERVE_CACHE`); `/raid` has 459 GB free, root
  161 GB.
- Prebuilt wheels only for torch (2.13.0+cu130) and FlashInfer (nightly
  `flashinfer-python` + `-cubin` + `-jit-cache` from flashinfer.ai/whl when a
  FlashInfer path is tested); never build FlashInfer from source or let it JIT
  a large kernel set (it exhausts host RAM and has crashed the box).
- Native builds capped: `MAX_JOBS=8 NVCC_THREADS=2` under
  `systemd-run --user --scope -p MemoryMax=120G`; serving under 150G;
  earlyoom is active.
- Docker only for the image-only B12X control; containerd root is on /raid.
- Never `pkill -f` a pattern the calling shell contains; kill by PID.
- Never A/B a compiled path with a private env var; use `graph_factors` or
  distinct code.
- Rebuild `_quixicore_C` after any merge that touches `csrc/`; swap the
  `.so` by rename, never overwrite a mapped file.
- Commit as Auroter, no trailers; push to `upstream` (QuixiAI) branch
  `glm53f-rtx6000` when the operator says so; PRs target QuixiAI main.

## 11. Decisions (operator answers recorded 2026-09-11)

1. Salvage: yes, the KEEP set of section 1d, one commit at a time,
   re-measured. Nothing else from the closed PRs.
2. Driver upgrade for an unhandicapped B12X control: still open; it only
   changes how honest the external comparison is, not our stack.
3. Speculator: whichever is fastest on the measured workloads. The DFlash2
   drafter's cc-by-nc-nd license is acceptable for self-hosted and internal
   use; the campaign may end with two records (DFlash2 for the fastest,
   the checkpoint's own MTP head as the commercially clean alternative).
   Phase 3 measures both on the prose harness and the structured 8-wide
   workload.
4. KV tier (explained, default chosen): SlimServe's HostTierConnector keeps
   evicted KV blocks in pinned host RAM and then on disk so long
   conversations and shared prefixes restore instead of re-prefilling. The
   A100 records carry 72 GiB per rank of pinned host RAM plus 256 GiB per
   rank of disk, which does not transfer to a 188 GB host with four ranks.
   Default for the `rtx6000` record: 32 GiB pinned per rank (128 GiB total)
   plus a disk tier under `/raid` sized to what is free at Phase 6. It is a
   completeness item, not a throughput item.
5. Cleanup: done for the local branch, the fetched PR refs and 32 GB of
   scratch build products, traces and stale compile cache. Two deletions on
   `/raid` were blocked by the tool sandbox and are left to the operator:
   `/raid/scratch/slimserve-glm53` (a 148 KB CMake cache) and
   `/raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4` (a 7.2 GB duplicate of
   the sidecars that live next to the checkpoint, verified byte-identical,
   plus symlinks).
6. NEW, needs an answer before Phase 6: hosting of the FP8 hybrid
   artifacts. The retained record depends on two sidecars built from the
   306 GB native checkpoint (`f32-overrides.safetensors` 1.2 MB,
   `fp8-swapset.safetensors` 7.2 GB). For anyone else to run the profile,
   `slimserve.fetch` must be able to download them: publish the sidecars in
   a SlimServe Hugging Face repo (recommended, small), or publish the full
   hybrid checkpoint (~200 GB). Until then the record is reproducible only
   with both checkpoints on disk and the builders.

## 12. Next command

```bash
cd ~/Lazarus/SlimServe && git branch --show-current   # glm53f-rtx6000
bash ~/.local/scratch/slimserve-glm53/rebuild.sh       # full 12.0f build of this tree (73 min; GPUs idle)
```

Then Phase 0 items 2-5, in that order, each with its notebook entry.
