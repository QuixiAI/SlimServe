# Qwen3.8-27B + DFlash 2 Metal Serving Handoff

Updated: 2026-08-23 00:58 (M5 Max MacBook Pro, 128 GB, ~460 GB/s measured
stream). Written so a fresh agent can take over cold. Read this, then
`perf/optimization_status.md` entries (19) onward, then
`perf/qwen38_metal_design.md` (every verified tensor map + mechanism).

## Mission and hard rules

- Serve profile `qwen38-q2kxl-1` on Metal: Qwen3.8-27B (unsloth UD-Q2_K_XL
  GGUF, 64-layer hybrid: 48 gated-deltanet linear-attention layers + 16
  full-attention layers, head_dim 256, interleaved MRoPE; plus the
  mmproj-F16 `qwen3vl_merger` vision tower) speculated by the Inco AI
  DFlash 2 drafter (z-lab Q4_K_M GGUF, block 8, top-16 path selector,
  two-tap convs).
- Bars: llama.cpp plain decode on this box/artifact = **35.67 tok/s**.
  Vendor DFlash 2 acceptance 4.80 is a GSM8K number; the llama.cpp
  dflash2-pr branch on the SAME GGUFs/settings gets 2.51 tok/step on our
  essay prompt and 4.74 on a GSM8K-style prompt (task-domain dependence;
  compare acceptance only on matched arms).
- **Speculation is always on and must be net-positive** (memory
  `spec-always-fastest`); slower-than-plain spec is a BUG, never a
  documented config. This gate is now closed: registered DFlash k=3 beats
  plain on both retained prompt arms and the matched exact-server workload.
- **Greedy / temperature 0 is banned stack-wide** (memory
  `no-greedy-benchmarks`; the user removed the flag on purpose). Validation
  and benches use the model's shipped sampling defaults from the GGUF
  (`general.sampling`: temp 1.0 / top_p 0.95 / top_k 20), seeded (42).
  Layer-level parity (cosine on activations) needs no sampling and stays
  the correctness instrument.
- Commit authorship: Eric Hartford sole author, no assistance trailers
  (the repo's signoff hook adds his Signed-off-by). Commit with
  `env SKIP=markdownlint-cli2 git commit ...` (the notebook's pre-existing
  line lengths fail markdownlint; its auto-fix also corrupts `+ ~15`-style
  lines and `_foo` identifiers -- never let it run on perf/).

## State of the tree

Committed on `main` (pushed): `fe960935f` "vision, DFlash 2 spec e2e,
native IQ decode, hybrid-pool layout fix (15 tok/s plain)" on top of
`39efaa7d9` (correct plain decode, layer parity). Prior campaigns: Muse
`ad8e8e937` (20.1 tok/s spec, Metal), DSV4 A100 `bad7cfd46` (A100 box).

UNCOMMITTED in the worktree is one tested optimization stack (preserve all
of it; do not treat the native pieces as abandoned experiments):

1. **Fused target GDN, complete and routed.**
   `csrc/quixicore/metal/kernels/serving_glue/gdn_step.metal`, the binding
   in `qc_metal_serving.mm`, `vllm/quixicore/ops.py`, and
   `qwen_gdn_linear_attn.py` implement decode and multi-position verify
   (convolution + recurrent scan, fp32 state in place, exact store/resume /
   rollback slots) plus a fused gated RMS norm. The torch-native oracle
   remains intact. Kill switch: `VLLM_QWEN38_FUSED_GDN=0`.
   Correctness: 147/147 exhaustive cases, all uniform/ragged/null/mixed
   plan cases, all gated-norm cases. Durable real-geometry tests are in
   `tests/model_executor/test_qwen_gdn_metal.py`.
2. **Verify-band quant MM, complete and routed.**
   `dequant.metal`, `qgemv.metal`, and the binding admit the target's
   Q2_K/Q3_K/IQ1/IQ2/IQ3/IQ4_XS formats to M={2,4,8,16,17} MM instead of
   repeated GEMV. Real-GGUF M=8/17 error <=0.2674%; sampled M=8 kernels
   are 2.4-5.0x faster than eight GEMVs. The eight-wide IQ decoder's
   difference from the scalar decoder is rounding-only (<0.1%).
3. **Seeded/vectorized MPS rejection, complete and routed.**
   `rejection_sampler_utils.py`, `qwen3_dflash2.py`, and `speculator.py`
   key all selector/accept/residual/bonus draws by (seed, position), remove
   the draft-logit double temperature divide, and batch rejection fully on
   MPS. Full-vocabulary Gumbel emission is now one keyed uniform plus CDF
   inverse sampling. Monte Carlo passes; the clean spec bench is seed-stable.
4. **Fused DFlash 2 convolution, built and routed.**
   `csrc/quixicore/metal/kernels/serving_glue/dflash2_conv.metal` replaces
   each repeat_interleave/roll/clone/elementwise graph with one dispatch.
   Both sides pass the torch reference at real 5120-hidden / 320-group BF16
   geometry; a same-process real-geometry microbench is 5.65x faster.
   Kill switch: `VLLM_QWEN38_FUSED_DFLASH2_CONV=0`. Powered end-to-end A/B
   retained it: fused essay/GSM medians 16.42/38.12 versus 15.75/36.22.
5. **64-bit hybrid KV gather, complete and routed.** MPS `index_select`
   silently wrapped signed 32-bit element offsets on Qwen's interleaved,
   strided K/V source. Requests crossing physical block 1271 therefore fed
   bad K/V into layer 19 and produced all-NaN target logits. The native
   `kv_cache_gather_range` carries the physical block stride and address math
   in 64 bits, gathers only live rows, and is exact at blocks 1186/1271/1580.
   A 20x64-token repeated run remains finite and seed-identical through the
   old failure window and allocator wrap.
6. **Profile/tests/build.** `qwen38-q2kxl-1` is supported, registers DFlash
   k=3, and exports `VLLM_USE_V2_MODEL_RUNNER=1`. The real server passed text
   and image. The final focused suite (including the 5 GiB >2^31-offset case)
   is 17/17; `tests/slimserve` is 58 passed/1 skipped. Final metallib SHA-256:
   `539035eb15dea29152e11503fc1ee08676d5dfe08b9ef4cc241283092e887d4c`;
   deployed extension SHA-256:
   `ede784f0d4ecf7a5111fc55374987661ea3bcc4c48602189c0a009cb88c4efdb`.
7. **Shared live validation.** Registry discovery finds `dsv4-xxs-1`,
   `muse-kdyn-1`, and Qwen on this machine. Qwen passes text+image. Muse now
   passes text+image after fixing its parser's new-turn state and the real
   split `" to"` / `"=self<|message|>"` streamed header; raw SSE cleanly
   separates `reasoning_content` and final `content`. The combined parser and
   SlimServe suite is 62 passed/1 skipped. The complete matrix is not green:
   DSV4 reached health but its first request ran at about 0.1 tok/s with 0/5
   drafted tokens accepted and was terminated after about 12 minutes.

**CURRENT STATUS:** no correctness or profile gate is blocking Qwen serving.
The old 20 W power blocker is closed; the retained numbers below were captured
on AC power after the charger change. Qwen's remaining gap is performance
versus the 35.67 tok/s llama.cpp plain reference, not production-path
correctness. Separately, the current-machine profile matrix is blocked by the
DSV4 Metal regression described below; do not present that shared matrix as a
pass.

## Measured numbers (seeded shipped defaults, V2 runner, in-process)

| Build | Plain essay | Plain GSM8K | Spec essay | Spec GSM8K |
| --- | ---: | ---: | ---: | ---: |
| campaign start (V1) | 2.5 | -- | -- | -- |
| V2 runner, fp16 dequants | 6.4 | -- | 4.0 | -- |
| + layout fix (strided gather penalty) | 2.2 | 2.0 | 1.0 | 2.2 |
| **fe960935f** (+ gather fix, native IQ) | **15.0** | **14.0** | 3.5-4.1 | 8.0-9.5 |
| + fused GDN/MM/vector rejection (pre-conv) | 16.15-16.40 | 15.81-16.11 | **15.03-15.60** | **36.06-36.66** |
| + powered stack, k=7 | 16.99-17.17 | 16.77-16.86 | 16.05-16.35 | 36.82-38.19 |
| **supported profile, k=3** | **16.99-17.17** | **16.77-16.86** | **23.06-23.74** | **34.33-35.25** |

Acceptance (Prometheus counters, essay, notebook (25)): 2.71 tokens/step,
0.244 draft rate -- beats the llama.cpp dflash2-pr branch (2.51/0.219).
Correctness: all 64 layers cos >= 0.9997 vs llama.cpp eval-callback;
corruption gauntlet 7/7 clean boots, 24/24 same-seed pairs identical;
fused GDN exhaustive harness 147/147 plus all plan shapes; rejection Monte
Carlo PASS; current focused durable suite 17/17; vision tower ~1e-3 vs
llama-mtmd-cli; real SlimServe text and image requests pass; Muse-Glimmer was
unregressed by a fresh profile-exact text+image smoke after its reasoning
parser repair.

The old 8-position x 48-layer Python GDN inversion and the >2^31 hybrid-cache
corruption are closed. At k=3, M=4 verification is a better Metal operating
point than the trained/upstream k=7/M=8 width: essay median rises from 16.18 to
23.27 tok/s while GSM remains 34.78. The exact registered server produced 128
input + 256 output tokens at 18.646 tok/s spec versus 15.913 plain (+17.2%).

## Open bugs / items, ranked

1. Remaining perf versus llama.cpp: Q4_K GEMV measures only ~95 GB/s on the
   5120x6144
   ssm_out shape (4.7x off floor, pre-existing); head_dim-256 paged
   attention fast path (the 16 full layers run SDPA; paged path is
   64/128 only); selector walk and the other five-layer drafter graphs.
   The k=3 top ledger is target 70.19 ms and inclusive sample/propose tail
   16.42 ms per step, so target bandwidth is again the primary wall.
2. Fix the DSV4 Metal profile regression exposed by the attempted complete
   live-smoke matrix. `dsv4-xxs-1` loaded 93.63 GiB and reached health, then
   spent about 12 minutes on the first tiny request at ~0.1 tok/s; the first
   draft had 0/5 accepted. This is grossly inconsistent with its historical
   33.684 tok/s baseline and must be isolated at first-step/verify granularity.
   Qwen and Muse both pass their registered text+image arms, but the full
   three-profile matrix remains failed until DSV4 completes.
3. Cosmetics/hygiene: env-gated diagnostics remain in
   `models/qwen3_5.py` (`_Qwen38DumpState`, layer-parity instrument) and
   `qwen3_dflash2.py` (`QWEN38_DFLASH_DUMP` recall@k dump) -- zero-cost
   unset; remove at campaign close. Stale `autostash` entry in `git
   stash` is from Aug 7 (DSV4 era), safe to drop. A stray token quirk
   appeared in sampled answers at temp 1.0 (both text and vision) --
   unattributed, low priority.

## Scripts and raw artifacts

Durable copies are under `perf/results/2026-08-22/qwen38-fused-gdn/`:
`consolidated_bench.py {plain|spec}` (essay + GSM8K arms, 3 seeded repeats,
prints BENCH_JSON), `collapse_probe.py`, `spec_profile.py`, and
`spec_profile_top.py`. Patterns: in-process `vllm.LLM(**engine_kwargs)`
from `slimserve.registry.resolve("qwen38-q2kxl-1","metal",1,None,2**37)`,
`__main__` guard (EngineCore spawns), `SamplingParams(temperature=1.0,
top_p=0.95, top_k=20, seed=42)`, `max_model_len` 8192 +
`gpu_memory_utilization` 0.45-0.6 for qwen38 (Muse smokes must use
PROFILE-EXACT kwargs -- a max_model_len override breaks its image
profiling).

Final raw data is under `perf/results/2026-08-23/qwen38-kv-gather/`:
`run_summary.json`, `exact_spec.json`, `exact_plain.json`, `smoke.json`, and
the real-server log. Shared-profile evidence is in `smoke-muse-final.json`,
`smoke-muse-final/muse-kdyn-1.log`, and `smoke-all/dsv4-xxs-1.log`. The exact
server harness command uses
`benchmarks/benchmark_dsv4_exact.py` with explicit `--temperature 1.0
--top-p 0.95 --top-k 20 --seed 42`; never rely on that harness's legacy greedy
default for Qwen.

## Ops gotchas (each cost real time)

- The Mac SLEEPS and kills background runs/agents: hold it awake
  (`mcp adrafinil keep_awake`, lid-closed included) for any long run.
- A 20 W charger at 1% battery throttles this workload catastrophically.
  Require a high-wattage supply and battery reserve before any baseline or
  phase-profile run; check `pmset -g batt` and the charger wattage first.
- Refreshing `vllm/quixicore_metal.metallib` / `_quixicore_C...so`: rm-then-cp
  + `codesign -f -s - <so>`; cp over the mapped inode SIGKILLs on dlopen.
- llama.cpp builds: `env -u LDFLAGS -u CPPFLAGS` (a custom-LLVM env poisons
  links); `~/llama.cpp/build-qwen38` (master, plain oracle),
  `~/llama.cpp-dflash2/build` (PR #27342 spec oracle); check a binary is
  fresh before trusting it (`strings ... | grep` a known-new symbol).
- Registry bytes/sha come from `curl -I` / the HF paths-info API, never
  from summarized pages (a wrong byte count masqueraded as a broken
  download for an hour).
- Regression smokes use profile-exact engine kwargs.

## Reference facts (verified; full maps in perf/qwen38_metal_design.md)

- GGUF arch strings: target `qwen35`, drafter `dflash` (three-way probe:
  `dflash.expert_count` -> DSV4 DSpark, `dflash.selector_rank` -> DFlash
  2, neither -> Muse) across config parser, tokenizer registry, loader.
- llama.cpp converter conventions undone at load: +1 fold in every norm
  weight except `linear_attn.norm` (we use GemmaRMSNorm -> subtract 1),
  GDN per-V-head tensors in TILED order (pairing i_k = i_hv % 16, cfg
  `gdn_tiled_v_head_layout`; the FLA Triton kernels still assume grouped
  if this GGUF ever runs on CUDA), `ssm_a` stored as -exp(A_log)
  (A_log = log(-ssm_a)), conv1d (dim,kernel)->(dim,1,kernel), MTP block
  `blk.64.*` unmapped, fused `attn_qkv` row-split into q/k/v shards
  (GGUF quantizes per output row), full-attn gate fused inside `attn_q`
  (per-head [q|gate], matches the vendored split).
- Hybrid shared block pool: attention views are restrided blocks-first
  (attn_utils `_update_hybrid_attention_mamba_layout`); metal_attn's SDPA
  path uses the native 64-bit range gather. Do not restore MPS `index_select`
  on the strided pages view: beyond 2^31 elements it silently reads the wrong
  address, even though its small-cache microbench is fast.
- Quant formats: all native on Metal (qgemv + qgemm tiles incl. IQ1_S,
  IQ1_M, IQ2_XS, IQ2_S, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL);
  `_DEQUANT_TYPES` is empty; only the Q2_K embed table dequantizes.
- Drafter: 5 layers all NON-causal (`dflash.attention.causal=False`),
  block 8 counts the anchor (7 drafted), target layers [5,19,33,47,61]
  0-based, selector A/B tables {248320,256} Q4_K dequantized at load. The
  registered Metal serving depth is deliberately k=3 after the powered sweep.
