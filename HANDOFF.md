# Muse-Glimmer-30B Metal Speculative Serving Handoff

Updated: 2026-08-15 (M5 Max MacBook Pro, 128 GB, ~460 GB/s measured stream)

MISSION (user, 2026-08-15): we are writing an inference engine and custom
kernels to serve Meta's published Muse-Glimmer + DFlash drafter AS-IS;
**the goal is to exceed llama.cpp's performance** (this box: 26.75 plain /
30.8 spec, pre-PR-26842), then chase the vendor's ExecuTorch numbers
(26.6 / 50.2 on M5 Max, same drafter). NO drafter training -- that
workstream is closed (notebook (39)). The standing 100 tok/s stretch
target remains; the measured route runs through step time: 30.8 needs
~88 ms/step, 50.2 needs ~53 ms (we are at ~127).

Prior handoff (DeepSeek V4 0731 A100 campaign: degeneration incident
RESOLVED; Q4_K fused MoE + cp.async landed) is preserved at commit
`bad7cfd46`; that campaign lives on the A100 box in
`/home/ubuntu/SlimServe`. This file covers the active local campaign.

## Mission and hard constraints

- Target: **100+ tok/s single-stream** for `muse-kdyn-1` (user mandate,
  judged fair for this hardware).
- **Speculative decoding is ALWAYS ON.** SlimServe is opinionated and ships
  the fastest configuration. STATUS: spec now BEATS plain decode
  (16.1-16.3 vs 14.4 tok/s) as of the tensor-ops verify kernels; the stale
  `--no-spec` note in `slimserve/profiles.json` has been removed and the
  spec numbers are the profile baseline (`perf/baseline_status.md`).

## The 100 tok/s ledger (physics, do not re-derive)

**2026-08-15 REVISION (measured, notebook (9)): the tree-speculation
multiplier is capped at ~3.0 tokens/step for buildable trees (upper-bound
replay of logged drafter top-8 across 234 steps; even a 611k-node tree
only reaches 3.47). The drafter has no signal beyond position ~3
(acceptance 0.77/0.50/0.19/0.11). Therefore 100 tok/s would need a step
below the 38.6 ms weight-pass floor -- NOT REACHABLE with this
model+drafter artifact. Honest serving-side ceiling: ~50-62 tok/s
(floor-adjacent step x 2.7-3.0). The drafter-training workstream is
CLOSED per user direction (notebook (39)) -- we serve the published
artifact AS-IS; the acceptance analysis above stands as the measured
property of that artifact, and the vendor's own 50.2 tok/s (ExecuTorch,
M5 Max, same drafter) is the serving bar. The original ledger below is
retained for context.**

- One weight pass = 19.6 GB / ~460 GB/s = **~43 ms floor per step** →
  ~23 steps/s ceiling for ANY step, plain or verify.
- Therefore 100 tok/s needs BOTH:
  1. verify step near the floor (was ~252 ms; now ~170 ms; target < 60 ms),
  2. **~4.5-5 accepted tokens/step** (currently 2.87) — tree /
     multi-candidate drafting, still unstarted. UNBLOCKED: the tensor-op
     rate probe measured M_PAD=48 at only +76% vs 32 (M nearly free on the
     M5 neural accelerators), so a 32-48-node draft tree costs about one
     weight pass (needs tree attention masks + multi-candidate DFlash
     blocks; study llama.cpp `common/speculative.cpp` for drafter block
     mechanics).
- Reference points: RTX 5090 (card): 74.9 plain / 233.4 spec. llama.cpp on
  THIS machine: 26.75 plain; best spec config 30.8.

## Current state (ALL UNCOMMITTED in this worktree)

Numbers (256-token greedy essay bench, port 8078, `muse_bench.py` under
perf/results/2026-08-11/muse-optimization-pass-1/):

- Spec-on CANONICAL RESTED BASELINE: **19.94 / 20.14 / 19.74 tok/s**
  short-context (fused verify + tensor kernels + winning route; rested
  45-min protocol, notebook (16); re-confirmed 19.97/19.97/19.86 by the
  (34) capstone on the final build). LONG-CONTEXT (10k, cached decode):
  **10.97 / 11.00** -- correct global attention + MQ kernel, +21% over
  the (broken) pre-fix state. Progression: 8.6 campaign start → 11.2 → 16.26 (tensor
  kernels) → 17.84 (device-X v15) → 18.26 (u4, observed midday) → 20.14
  (fused verify re-flip). The FUSED VERIFY IS NOW THE DEFAULT
  (VLLM_MUSE_FUSED_VERIFY=0 reverts); its matmuls run the eager route's
  variant table inside emit_matvec. Plain fused decode: 14.4-16.4.
- Acceptance: 1.63-1.79/draft (healthy band; greedy trajectories shift
  slightly across kernel variants from fp accumulation order).
  **Acceptance is NOT broken; do not chase it.**
- Verify GEMMs (M=17, device-X tensor variant 15): o_proj Q6_K 174 µs,
  attn_gate Q5_K 89.5 µs (2.2× its floor), ffn_gate Q5_K 353 µs (1.78×),
  ffn_down Q4_K 370 µs (2.28×).

## The measured step ledger (2026-08-15, verify step ~153 ms in-process)

- Target forward M=17: **119 ms** (matmul floor 36.6 ms + attn/glue ~20)
- Drafter propose: **23.7 ms** (model fwd 12.6 — 3.6× its 3.5 ms floor,
  dispatch-bound, never got the muse_step fusion; sample_draft 5.9; prep
  glue 5.2). NOTE: drafter runs OUTSIDE execute_model in this fork.
- compute_logits 4.8 + rejection sampler 3.3 + engine residue ~10.
- Serving through the API stack costs ~10 ms/step more than in-process.
- Plain decode (M=1): 61 ms in-process; GEMVs measured at 1.10× floor
  overall (big shapes 0.88-1.03× — kernels healthy); llama.cpp does the
  whole plain step in 37.4 ms = AT the 18.9 GB/step bandwidth floor.
- 100 tok/s = ~4.75 tok/step (tree) × ~48 ms step. Every leg above has
  measured headroom; nothing is at floor yet except the M=1 big GEMVs.

## What was measured (2026-08-14, replaces all inference-from-deltas)

Stage-ablation profile of the 4-warp paired kernel on the down shape
(variants 21..27, bench-only, `perf/results/2026-08-14/qgemm-sm-profile/`):

- Full simdgroup p4 = 634 µs (3.90× the 163 µs floor). Exposed per-stage:
  **MMA issue 290 µs** (dominant); weight stream+decode 52; dequant ALU 25;
  X re-staging 21; barriers 18; fragment loads 14. Staging-only = 331 µs =
  2.04× floor; pure weight-stream = 223 µs = 1.37× floor.
- Standalone matrix-pipe rate probe (probe.metal/probe.mm, same dir):
  chained simdgroup MMA 6.4 TFLOPS vs **mpp::tensor_ops::matmul2d 55
  TFLOPS** at the exact production tile shape (M5 GPU neural
  accelerators). M_PAD 24 vs 32 free; 48 only +76%.

## What was built (all measured, parity-verified)

1. **qgemm_sm_t / qgemm_sm_t2** (variants 14/15; 15 is the production
   verify route): qgemm_sm_p weight staging (paired-plane dequant
   q4_K/q5_K, span q6_K, BK=64, double-buffered, one barrier/step) + one
   cooperative 32×64 @ 64×32 `tensor_ops::matmul2d` per K-step
   (`execution_simdgroups<4>`, multiply_accumulate into a float
   cooperative_tensor, `cT.store` to the same (4, N, 32) float partials;
   deterministic reduce unchanged). v15 additionally skips X staging and
   hands matmul2d a device tensor slice per K-step (X is L2-resident;
   measured attn_gate −40%, ffn −13..15% over v14). Route in
   `vllm/quixicore/ops.py::ggml_mul_mat_sm`: Q4_K/Q5_K/Q6_K, K%64==0,
   N%32==0 → 15; simdgroup variants 9-13 remain as fallback.
   Metallib now builds at **-std=metal4.0** (`cmake/metal.cmake`); all 81
   pre-existing kernel sources compile unchanged.
2. Stage-ablation bench variants 21..27 (`qgemm_sm_pa`, q4_K only) +
   round-robin min-of-rounds profile script — rerun after any staging
   change instead of guessing.
3. Earlier this campaign (see notebook 08-11..14): qgemm_sm family
   (variants 9-13), paired-plane dequant, fused M=17 verify
   (muse_step_run_aux, GATED OFF — see below), vectorized k-quant loads.

## Pitfalls already paid for (do not repeat)

- The cooperative_tensor doc-comment API in MPPTensorOpsMatMul2d.h is
  stale: use `is_valid_element(i)`, not `get_mask(i)`; `#pragma unroll
  full` is not a thing (use `#pragma unroll(N)` or nothing).
- Tensor A/B tiles: extent(0) is the CONTIGUOUS dim (x = columns). A
  32×64 row-major tile is `extents<int32_t, 64, 32>`. The four contiguous
  8×64 per-warp st slices form the 32×64 A tile with no staging changes
  (st is plain row-major; swizzle in st.metal is commented out).
- The fused verify is DEFAULT ON since 2026-08-15 (notebook (15)-(16)):
  its emit_matvec now mirrors the eager variant table; fused forward
  95.6 ms vs eager 110.4 at M=17, e2e +13% rested. Keep the two variant
  tables in sync when routes change.
- llama.cpp muse-pr branch: `llama-cli` REPL-loops forever; use
  `llama-server` + `/completion`, pass `--spec-type draft-dflash`.
- Bench triplets thermally decline (16.3 → 14.6 within ~90 s sustained).
  Compare matched positions across configs, or wait for cool-down.
- `qgemv_mm` host-side instantiations are only M ∈ {2,4,8,16,17}; other M
  values crash if routed.
- Metal has no lambdas; `as_type` is a bare builtin; plain (non-host_name)
  kernels in `namespace mittens` need the `mittens::` pipeline prefix.

## Next steps, in order (sized by the measured ledger)

0. **LONG-CONTEXT DECODE (notebooks (26)-(28)): cached-prompt decode
   falls 20.25 -> 16.1 -> ~9.1 tok/s at ~330 / 1.9k / 9.9k ctx, and the
   loss is SHARED by both attention routes (expansion vs SDPA a wash at
   every length -- the 17x-re-read theory is dead, and raw KV bytes only
   explain ~4.5 ms of the +167 ms/step at 10k). ATTRIBUTED (notebook (29)): the
   entire loss is the two forwards' attention -- target +110 ms/step and
   drafter +10 at 9.9k ctx, per-row scan ~25x off the KV stream floor;
   everything else is flat. KERNEL BUILT AND PROVEN (notebook (30));
   debugging its route EXPOSED AND FIXED an eager long-context
   correctness bug (notebook (31): Attention()'s cache_config fallback
   window-clamped the 13 global NoPE layers at 2048 -- acceptance at 10k
   recovered 2.06 -> 2.78 tok/step with the fix; any pre-fused-era eager
   deployment beyond 2k ctx served degraded long-range attention). MQ
   kernel now engages in eager (log-confirmed; 10.19 tok/s at 10k).
   Fused stays default (always-correct, still fastest). REMAINING: plumb
   paged_attention_verify into the fused encoder's global-layer site
   (ctx_len arg + launch swap + partials scratch; projected fused fwd at
   10k: 207.8 -> ~145 ms). Bench discipline: decode
   numbers from cached-prompt repeat runs only; VLLM_EXPAND_CTX_MAX
   stays no-op.**

1. **uint4/int8-native verify matmuls** — q4_K SHIPPED 2026-08-15 (see
   notebook (4): qgemm_sm_u4 + muse_u4_repack + GGUFLinearMethod route,
   ffn_down -15% matched, e2e 18.26 flat; kill switch VLLM_MUSE_U4=0).
   REMAINING: q6_K int8-native was built and measured NEGATIVE (~15%
   slower than v15 on o_proj -- small-K shapes are fixed-cost-dominated,
   not stream-dominated; kernel kept bench-only, notebook (5)). The
   small-shape lever is fixed-cost reduction (fold the reduce into the
   GEMM kernel, split_k tuning, fused-verify encoder), plus fold the
   ~120 us of xsum/reduce/encode overhead out of the u4 op, then q5_K
   staged BK=128 for narrow shapes is DONE (variant 17: attn_gate -31%,
   notebook (8); wide gate/up stays on v16 -- BK=128 loses there). The
   wide-shape q5_K stream: the split-K partials machinery was the shared
   fixed cost (t2w8 ablations, notebook (10)); variant 31 (split-K=1, one
   float slice + SK=1 cast; NEVER scatter coop-tensor elements manually --
   the w8 layout holds partial sums, only cT.store is safe, notebook (11))
   is parity-exact and matched-won +33-44% HOT -- but LOST the
   same-protocol rested A/B e2e (16.05 flat vs legacy 17.81; notebook
   (13)) and is now opt-in via VLLM_SM_ROUTE=split1. Default route:
   15/16/17 + u4 intercept (A/B winner). METHODOLOGY: hot-matched
   microbenches nominate candidates; route flips require the rested
   45-min A/B protocol (scratchpad cool_bench scripts). Canonical rested
   baseline: 17.81/17.56/17.58.
   Original probe
   evidence:
   u4_scaled runs the down shape at **193.7 us = 1.19x floor** (vs 370
   for v15), with per-32-group scale correction measured free and A =
   row-major X^T (no transpose). Probe sources archived in
   perf/results/2026-08-15/plain-step-decomp/probe2.{metal,mm} — the
   kernel structure there IS the implementation spec (tile-major packed
   B, tilek=32 multiply into tmp coop tensor, sc[n,g] fma into main).
   Implement: (a) load-time repack q4_K→uint4 + q6_K→int8 (tile-major
   [n_tile=64][K][NT] + scale/min planes; +1-26% bytes; load-time
   replacement, NO lazy caches); (b) qgemm_sm_u4/u8 kernel incl. the
   min-term rank-1 side GEMM (per-group X column sums); (c) q5_K
   (ffn_gate/up, lm_head) gets NO native win (int8 is a byte wash) —
   improve its staged path instead: BK=128 GEMV-style contiguous runs
   (the M=1 GEMV kernels prove this layout streams at 0.96x floor).
   Target verify forward 119 → ~70 ms.
2. **Drafter fusion** (~21 -> ~8 ms): the prep-glue syncs are DONE
   (native prepare_dflash_inputs Metal kernel, token-identical, notebook
   (7)); remaining are the model forward (12.6 ms vs 3.5 floor) and
   sample_draft (5.9 ms -- on-device argmax/Gumbel). ARCHITECTURE NOTES
   (2026-08-15 scouting): the drafter is NOT MuseGlimmer-classed -- it is
   MuseGlimmerDFlashDraftModel extending DFlashQwen3ForCausalLM
   (qwen3_dflash.py: 5 layers, per-layer causal/SWA config, fc encoder,
   q/k norms, non-causal attention), so muse_step's fused machinery does
   not transfer directly; a fused drafter step is a bespoke emit-chain
   against that structure. Also inside the 21 ms:
   MuseGlimmerDFlashModel.precompute_and_store_context_kv runs PLAIN
   TORCH per step (per-layer qkv_proj + rms_norm + rope + cache writes
   across 5 layers -- a block of small dispatches worth folding into any
   fused step). Its linears already route through the tensor kernels via
   the eager sm route. FIXED AND VALIDATED 2026-08-15 (notebook (22)-(23)): the fused
   drafter is parity-exact (per-layer 7-11e-3, forward 7e-3),
   token-identical in production -- the bug was CAUSAL semantics (this
   fork's DFlash SWA layers are causal; use the verify path's per-row
   base+i+1 expansion, and seq_lens must be CONTIGUOUS). Rested A/B: a
   WASH (5 layers carry too little dispatch), so the default stays OFF
   (VLLM_MUSE_FUSED_DRAFTER=1 to enable). Next drafter win lives in
   fusing sample_draft + ctx-KV precompute into the same command buffer
   on this now-proven path.
3. **Tail + engine** (~18 ms in-process + ~10 serving): on-device greedy
   rejection kernel, tensor-ops qgemm_sm_rm then re-test the fused-verify
   flip, async scheduling / encode-ahead for the engine residue, API-path
   overhead audit.
4. **Tree speculation** (multi-candidate DFlash blocks + tree-mask
   verify): 2.7 -> ~4.5-4.75 accepted/step. The largest multiplier; M is
   measured-free up to 48 on the tensor units. Batch expansion won't
   survive M=48 at long context (KV re-read per virtual row) -- needs a
   real tree-mask attention path.
5. Keep the ablation profile + rate probe in the loop: re-measure after
   each change, notebook every experiment. Bump the ggml_mul_mat_sm
   TORCH_CHECK variant ceiling with every new variant (v15 crash paid).

## Build / bench / validate

- Build (metallib + host): `cmake --build build/temp.macosx-11.0-arm64-cpython-312`
  then copy `quixicore_metal.metallib` and `_quixicore_C.cpython-312-darwin.so`
  from that build dir into `vllm/`.
- Kernel microbench: `perf/results/2026-08-13/qgemm-sm/qgemm_sm_bench.py`;
  ablation profile: `perf/results/2026-08-14/qgemm-sm-profile/profile_ablation.py`;
  MMA rate probe: same dir, `probe.metal` + `probe.mm` (compile with
  `xcrun -sdk macosx metal -std=metal4.0`).
- Serve: `.venv/bin/slimserve muse-kdyn-1 --serve -y --port 8078`.
- E2E bench: `perf/results/2026-08-11/muse-optimization-pass-1/muse_bench.py`.
  Acceptance: `curl :8078/metrics | grep spec_decode`.
- Every experiment goes in `perf/optimization_status.md`; raw logs under
  `perf/results/YYYY-MM-DD/<run-id>/`.

## Validation bar before claiming a win

Real profile through SlimServe (`muse-kdyn-1`), health + benchmark
workload + acceptance metrics unchanged (~1.73/draft) + coherent greedy
output, recorded TPS with raw artifacts. Interval logs and single warm
runs are diagnostics, not baselines. Preserve prior-agent worktree
changes — everything above is uncommitted.
