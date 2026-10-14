# qwen38fn-nvfp4-4 (Qwen3.8-Flash-Next NVFP4, 4x RTX 3090): optimization plan

Brainstorm of 2026-09-07, run per the QuixiCore CUDA handbook
(~/QuixiCore/QuixiCore-CUDA/perf/perf.md): inventory, references, baseline,
classify, then one-factor experiments kept only on a >=3% (low-risk) or
>=8-10% (complexity-adding) measured win with a counter-backed explanation.
Techniques were cross-checked against the Metal and ROCm sibling handbooks
(`perf/findings.md` in each) and this repo's own Metal campaign
(perf/metal_m1ultra_retrospective.md). Everything measured below is on this
box on 2026-09-07; everything estimated is labelled as such.

## 1. Where the profile stands (measured, production record)

| workload (exact 1000 in / 2000 out, seeded) | tok/s | notes |
|---|---:|---|
| c1 | 120.1 | ~15.9 ms per step at 1.9 tokens/step (MTP k=2, 51.6% acceptance) |
| c8 | 361.7 | only 5 requests run: each charges the packed slab ~18 rows |
| c16 | 335 | == c8: pool-bound, not compute-bound |

Reference points: the 8-GPU FP8 record does 131-136 / 547-585 / 838 / 880
at c1/c8/c16/c32 (twice the cards, half the weights per rank).

## 2. Host reality (measured 2026-09-07 on GPUs 4-7, idle)

- DRAM copy roofline: **844 GB/s** (bf16 1 GiB clone, read+write) vs 936 spec.
- NCCL all-reduce at TP4 (PCIe P2P driver, NCCL_P2P_LEVEL=SYS), bf16
  [t,2560]: t=1 41.7 us, t=3 48.5, t=8 40.2, t=24 42.5, t=64 77.1 us. It is
  a ~40 us latency floor up to ~64 tokens.
- vLLM custom all-reduce over PCIe (VLLM_CUSTOM_AR_ALLOW_PCIE=1) at TP4,
  car_parity.py: **t=1 24.0 vs 40.2 us, t=3 25.1 vs 40.2**, t=8 50.0 vs
  39.9, t=16 52.6 vs 40.6, t=64 174 vs 48, t=256 649 vs 127; parity PASS.
  The 2026-08-27 rejection was measured at TP8 (7 peers to read); at TP4
  the one-shot wins the decode-size payloads by 15 us.
- cuBLAS bf16 GEMM at decode M on this record's per-rank shapes (GB/s of
  weight bytes, M=3):

  | shape | us | GB/s | class |
  |---|---:|---:|---|
  | GDN in_proj 2560->4096 | 29.8 | 703 | near roofline (83%) |
  | GDN out_proj 1536->2560 | 16.9 | 466 | half roofline |
  | QSA qkv 2560->2048 | 19.7 | 531 | |
  | QSA o 1536->2560 | 16.6 | 474 | |
  | hyper mix_down 10240->320 | 20.6 | 318 | latency-shaped |
  | hyper mix_up 320->10240 | 16.5 | 397 | |
  | shared expert 2560->320 | 20.5 | **80** | pure launch/latency floor |
  | router gate 2560->512 | 21.0 | **125** | pure launch/latency floor |
  | lm_head 2560->62080 | 410 | 774 | fine |

  M=1/8/24 are within 10% of the M=3 row: these kernels are latency-shaped,
  cuBLAS picks tiles that leave most of the 82 SMs idle for N<=512, and a
  16-21 us floor per call applies no matter how few bytes move.

## 3. Serving-path inventory and bytes per step (per rank, TP4 + EP)

Checkpoint (ModelOpt MIXED_PRECISION): only the 512 routed experts are
NVFP4 (63.3 GiB, 2.6 MB per expert). Everything else is **bf16**: GDN
projections 3.89 GiB, hyper-connection mixers 1.19 GiB (nn.Linear,
**replicated on every rank**), QSA projections 1.15, lm_head 1.18,
embed 1.18, shared experts 0.44, router gates 0.12, vision 0.84. The PLE
n-gram table is fp8 in pinned host RAM (UVA gather).

Per decode step at c1 (3 query rows: 1 + k=2), per rank:

| stream | bytes | at 844 GB/s |
|---|---:|---:|
| routed experts, ~7.5 of 128 local experts x 48 layers x 2.6 MB | 0.93 GB | 1.1 ms |
| GDN projections (TP-split) | 0.97 GB | 1.15 ms |
| hyper-connection mixers (replicated) | 1.19 GB | 1.4 ms |
| lm_head slice | 0.30 GB | 0.35 ms |
| QSA projections | 0.29 GB | 0.35 ms |
| shared experts + router gates | 0.23 GB | 0.3 ms |
| **total weight stream** | **3.9 GB** | **4.6 ms** |

KV/state traffic at chat context is small: 36 GDN states x 786 KB
read+write, the indexer's compressed keys (dense scan, 64 B/token/layer),
and the top-2048 main-KV gather (12 layers x 2048 x 512 B) from GPU hot
rows or over PCIe.

Kernel launches per layer at decode (from the module structure): 2
all-reduces; 5-6 hyper-connection matmuls plus norms/sigmoid/gating; GDN
in/out projections plus the triton conv+scan+norm chain, or QSA qkv/o plus
indexer projections, scoring, top-k, gather, attention; router gate;
shared expert gate|up and down; Marlin routed MoE (align + GEMM + finalize).
Roughly 30-40 kernels per layer, ~1,700 per step, replayed from a FULL
decode CUDA graph (launch cost hidden; per-kernel GPU floor of 3-20 us not).

## 4. Step budget at c1 (measured parts + estimates) and the classification

| term | ms | basis | class |
|---|---:|---|---|
| 96 all-reduces | 4.0-4.6 | measured 40-48 us each | latency-bound collective |
| dense cuBLAS chain (~11 calls/layer x 48) | ~8-10 | measured per-call 16-30 us | latency-shaped skinny GEMM |
| routed experts (Marlin W4A16, M=1-3 rows/expert) | 1.5-3 (est.) | 0.93 GB at 300-600 GB/s | unmeasured: nsys owed |
| attention + indexer + GDN chain + glue | 1-2 (est.) | | many tiny kernels |
| host: scheduler, residency prepare_step (0.4 ms), sampler | 1-2 (est.) | | host floor |

The sum over-explains the 15.9 ms step, which is expected: the dense chain
and collectives are serial, and some cuBLAS calls overlap nothing. The
ranking is nonetheless robust: **at c1 the dense skinny-GEMM chain and the
collectives together are two thirds of the step, and neither is bandwidth-
bound.** The weight stream itself (4.6 ms at roofline) is a floor we are
far from; halving bytes (fp8 dense) only pays after the chain runs near
bandwidth. At c8 the limiter is different and already known: the packed
slab admits 5 requests, so throughput is 5 x per-request rate.

## 5. Experiments, ranked by expected recovery / effort

E0. **nsys decode census (measure before building).** Boot the record on
    GPUs 4-7 (port 8001), `nsys profile --trace=cuda,nvtx
    --cuda-graph-trace=node` over a c1 and a c8 window, rank kernels by
    total GPU time per step, count kernels per layer, read Marlin's
    achieved bytes/s on the expert GEMMs. This replaces every estimate in
    section 4 and is the gate for E2/E4. Half a day; no code.

E1. **Custom all-reduce for decode payloads at TP4** (config + a size
    cutover). Measured 24-25 vs 40 us at t<=3. Wire `VLLM_CUSTOM_AR_ALLOW_PCIE=1`
    into the record's env with the custom-AR size cap set so t<=3 payloads
    (<=16 KiB) go one-shot and larger ones fall back to NCCL (t>=8 loses).
    Expected: c1 -1.4 ms/step (~+9%), c8 neutral (its all-reduces are
    5x3=15 rows). Gate: exact bench c1/c8, byte-identical greedy outputs vs
    NCCL (the parity harness already shows CA is the more accurate sum).
    A day.

E2. **Skinny decode GEMM family for the bf16 dense chain** (the Blackwell
    `low_latency_gemm.py` route already exists in this fork for SM103; SM86
    has nothing). One weight-stationary kernel (Triton first, CUDA if Triton
    caps out) for M<=24: split-K across the grid to fill 82 SMs, 128-bit
    weight loads, fp32 accumulate, fused epilogues where the consumer is
    elementwise (sigmoid gate in the hyper-connection mix, SiLU*up in the
    shared expert, residual add). Targets: shared expert / router
    80-125 -> 500+ GB/s, the 1536->2560 and mixer shapes 320-470 -> 600+.
    Expected: dense chain 8-10 -> ~4-5 ms (c1 +25-35%), c8 proportionally
    less. Provenance: QuixiCore-CUDA `qgemv`/`qgemm_ksplit` (split-K at
    <~832 tiles on the 3090), Metal `qgemv_fused` (up+gate+act fusion
    1.4-3.2x) and its "weight-stationary small-M GEMV +15.6%" campaign win,
    ROCm "wave-per-output split-dot wins 5.5-79x for latency-shaped decode
    work and loses for GEMM-shaped tiles". Gate: microbench vs cuBLAS on
    the section-2 table first (an hour), then the model. One to two days.

E3. **Hyper-connection fusion** (after E2, same kernels). Each layer runs
    two GatedResidual blocks: norm -> mix_down -> SiLU -> mix_up -> sigmoid
    -> gate multiply -> combine (block_inject). Fuse per block into one or
    two launches with the mixer weights read once. Metal recovered 39 ms
    of 498 from exactly this fusion ("fused mHC"). Expected: -1 to -1.5
    ms/step at c1 on top of E2. Bit-exact class (fp32 accumulate, same
    order) so the A/B gate is identical logits.

E4. **Marlin NVFP4 decode geometry for EP tiny-M** (gated on E0). If the
    census shows the routed experts below ~400 GB/s: sweep Marlin's M<=8
    configs and thread-k/n choices, cut `moe_align` overhead, and if that
    is not enough, a QuixiCore-shaped fused route/align -> grouped W4A16
    GEMM -> SwiGLU -> finalize (the a100_glm52_design.md playbook; the
    QuixiCore-CUDA `moe_gemm_nvfp4` shows the dual-fp4 fragment decode and
    32-row M-blocking that gave 1.4-1.6x there). Expected: up to -1.5 ms.
    Two to four days if the fused kernel is needed.

E5. **Tokens per step (amortize the fixed ~6 ms of collectives + host).**
    (a) Speculation: k=3 via multi-length graph capture ("dynamic-k wins
    c1-c4 by 8-15%" in the FP8 backlog) - acceptance here is 51.6%, lower
    than the FP8 record's 58.9% because the block-FP8 drafter models an
    NVFP4 target; measure acceptance x cost before and after. (b) The pool:
    per-group slab strides so a chat request stops charging 12 GDN
    snapshots at attention-sized slots (5 -> 8 running at c8; the standing
    owed item from docs/host_resident_kv_design.md). Expected: c8 +40-60%,
    c1 +8-15% from (a). Config plus planner work; (b) is the biggest c8
    lever on the table.

E6. **fp8 weight-only for the bf16 dense layers** (after E2, so bytes
    matter). Mixers 1.19 -> 0.6 GB, GDN projections 0.97 -> 0.49, lm_head
    0.30 -> 0.15: -1.2 GB/step/rank = -1.4 ms at roofline. Marlin W8A16
    already serves block-fp8 on SM86 (the FP8 record's experts); the dense
    path needs the fp8 weight-only route enabled for the ModelOpt-ignored
    modules at load. Quality gate: deep recall + WildChat replay parity.
    Two days.

E7. **Host floor.** Residency `prepare_step` rebuilt every step (0.4 ms at
    TP4, 10.6 ms at TP8): make it incremental (device tables updated only
    for changed rows, pinned staging, no `.item()` in set_home/flush -
    the Metal "drain kill" lesson: every per-step blocking transfer is
    conserved until the last one dies). Then evaluate async scheduling
    (unset on the record) with spec decode + the tier connector. Expected:
    -0.5 to -1 ms at c1. A day.

E8. **GDN decode chain** (only with E0 evidence). The upstream fused GDN
    decode kernel was rejected on the FP8 record (-4.5% c8); the triton
    chain is 36 layers x several kernels. If E0 shows >1.5 ms in it,
    revisit with the Metal fused-GDN-step design (perf/qwen38_metal_design.md
    section "Fused GDN decode step") rather than the upstream kernel.

## 6. Do not redo (measured elsewhere)

- Custom all-reduce at **TP8** (loses at t>=3: 7 peers) - E1 is the TP4
  case, which measures differently. Pinned-host UVA one-shot all-reduce at
  decode sizes (cannot beat NCCL's ~40 us LL floor at 8 ranks).
- DP2 x TP4 (hyper-connection cannot shard by sequence; forced SP MoE).
- max_num_batched_tokens 1024 (-14% c8) and the vision image bound
  (over-commits the pool; OOM under c8) - 2026-09-06 notebook.
- Fused upstream GDN decode kernel (-4.5% c8 on the FP8 record).
- TurboQuant main KV on this record (operator: fp8, not TQ).
- Threadgroup/LDS staging of decoded weights for one-warp decode kernels
  (rejected three times on Metal, two-sided on ROCm: only pays with
  multi-wave reuse per barrier).

## 7a. Status log (2026-09-07 loop)

- E1 custom AR at TP4: REJECTED on step time (17.00 vs 16.81 ms).
- E0 census: done (notebook); c1 metric changed to ms/step (acceptance
  lottery), c8/c32 to verify-steps/s.
- E2 skinny GEMM route: RETAINED, on by default in the record
  (VLLM_QWEN4_EXP_SKINNY_GEMM=1): c1 -5.8% step, c8 +0.9%, c32 +3.5%.
  Needed the env registered as a compile-hash factor (cached graph kept
  cuBLAS) and the per-rank shapes read from the checkpoint (13 shapes).
- Next: NCCL protocol/algorithm sweep at decode payloads (collectives are
  39-80 us per call in-graph at c8), then E7 host floor with py-spy, then
  E5(b) per-group slab strides (c8/c32 capacity).

## 7. Execution order

E0 (census) and E1 (custom AR at TP4) first: both are measurement-heavy
and code-light, and E1 is the only item that needs no kernel. Then E2 ->
E3 -> E6 as one thread (the dense chain), with E5(b) in parallel on the
planner. E4 and E8 wait for E0's numbers. Every step records baseline,
hypothesis, correctness, exact-bench throughput and decision in
perf/optimization_status.md; tests run on port 8001 on GPUs 4-7 with
production untouched.
