# A100 Fast-Path Design: GLM-5.2-Vision GGUF

Design for serving `QuixiAI/GLM-5.2-Vision-GGUF` as fast as possible on
A100-SXM4-80GB (SM 8.0). Every number marked **[M]** was measured on this host
on 2026-07-30; everything else is derived arithmetic or a proposal.

Companion: `perf/perf.md` (method, host constants, rooflines).

---

## 0. The governing constraint

A100 is a **bandwidth machine with a tight vector-ALU budget**. The ratio that
decides every kernel decision here:

| host | achievable DRAM | FP32 | **FP32 : bandwidth** |
|---|--:|--:|--:|
| A100 (this host) | **1769 GB/s [M]** | 19.5 TFLOP/s | **9.6 FLOP/byte** |
| RTX 3090 (old host) | 936 GB/s | 35.6 TFLOP/s | 38 FLOP/byte |
| MI300X (the ROCm target) | ~5300 GB/s | ~163 TFLOP/s | 31 FLOP/byte |

A100 has **3–4× less vector compute per byte** than either machine this code was
tuned on. Two consequences drive the whole design:

1. **Bytes are the currency.** Never trade bytes for ALU work — no scale
   expansion, no dequant-to-fp16 round trips through DRAM, no format widening.
2. **But the ALU budget is not zero.** For Q2_K it is ~1.8 vector ops per weight.
   That is enough *if spent deliberately*, and the difference between a 12% kernel
   and a 94% kernel is entirely in how those ~1.8 ops are spent.

Per-format budget to remain DRAM-bound (at 1750 GB/s, 9.75 T int32 ops/s):

| format | bytes/weight | weights/s at roofline | **vector-op budget / weight** |
|---|--:|--:|--:|
| Q2_K (routed experts) | 0.328 | 5.34e12 | **1.83** |
| Q6_K | 0.82 | 2.13e12 | 4.57 |
| Q8_0 (dense/attention) | 1.0625 | 1.65e12 | 5.92 |

Tensor-core headroom, by contrast, is enormous: int8 IMMA measured **353 TOP/s
[M]** = 1.77e14 MAC/s against the 5.34e12 MAC/s that Q2_K at roofline demands —
**33× spare**. The MAC is free; only the unpack and the reduction cost anything.

---

## 1. Why Q2_K sits at 12–32% today

The fork's own MI300X profile (`SlimServe/perf_worklog.md:894`):

| path | % of DRAM peak |
|---|--:|
| MoE Q2_K `w13`, any batch | **32%** |
| MoE Q2_K `w2`, any batch | **11.9%** |
| shared-expert Q2_K `mmvq` | 8% |
| Q8_0 `mmvq` (best dense case) | 53–60% |
| every `mmq` tile path | 0.2–9% |

The pattern is exactly what the budget table predicts: **the more aggressively
quantized the format, the further from roofline.** A naive Q2_K inner loop costs
~5–7 ops/weight (extract 2-bit field, decode 4-bit sub-scale, decode 4-bit
sub-min, convert, multiply, subtract) against a 1.83 budget → a hard ceiling near
30–37%, which is precisely where `w13` sits. Q8_0 has 3× the byte budget and a
trivial dequant, so it reaches 53–60%.

**Porting that structure to A100 unchanged would be worse, not better**, because
the op budget is 3.2× tighter here than on MI300X.

---

## 2. The Q2_K fix

### 2.1 The factorization

Q2_K stores, per 256-weight superblock (84 B — `scales[16]`, `qs[64]`,
`half2 dm`): a 2-bit quant `q_i`, and per 16-element sub-block `j` a 4-bit scale
`sc_j` and 4-bit min `m_j`, both scaled by superblock fp16 `d`/`dmin`:

```
w_i = d·sc_j·q_i − dmin·m_j
```

So the dot product against activations `x` factorizes:

```
dot(w,x) = d · Σ_j sc_j·(q·x)_j  −  dmin · Σ_j m_j·(Σx)_j
                    ↑                           ↑
            int8 dot, per sub-block    activation-only: hoisted
```

Two structural wins fall out **in the integer domain**:

- **`(Σx)_j` does not depend on the weights at all** — computed once per activation
  tile, reusable across all 2048 output rows *and* all 256 experts.
- **`sc_j` and `m_j` are 4-bit and the sub-dots are bounded**, so the entire
  per-superblock reduction can stay in **int32 IMAD**. The only floating-point
  work is `d·A − dmin·B`: **2 float ops per 256 weights** = 0.008 ops/weight.

Two corrections to claims that look true but are not:

- **GGUF's existing activation sum is at the wrong granularity.**
  `block_q8_1.ds.y` does store a precomputed sum (`gguf_kernel.cu:37-42`), but over
  **32** elements (`QK8_1=32`), while Q2_K sub-blocks are **16**. It is consumed
  only by the q4_1/q5_1/q8_1 paths and is **unusable** here without extending the
  quantizer to emit 16-granular sums.
- **On a tensor-core kernel this factorization is unnecessary** — see §2.4. It is
  the right answer only for an int8-activation variant.

### 2.2 Measured validation [M]

Streaming 2.1 GB of Q2_K-layout blocks, 108×4 blocks/SM, 256 thr/block, runtime
(non-foldable) activation words:

| variant | GB/s | % of ceiling |
|---|--:|--:|
| pure byte stream (ceiling) | 1769 | 100% |
| 2-bit unpack + `dp4a`, single accumulator | 1600 | 90% |
| + float per-sub-block epilogue | 1453 | 82% |
| **+ integer epilogue (proposed)** | **1667** | **94%** |

Three findings, all actionable:

1. **The 2-bit→int8 spread is free.** `(w >> sh) & 0x03030303u` produces four
   int8 lanes in one lop3-class op — 0.25 ops per weight. No lookup table, no
   `prmt`, no divergence. This is the Marlin lesson applied to 2-bit.
2. **The integer epilogue *beats* the plain `dp4a` loop** (94% vs 90%). Sixteen
   independent sub-dots break the single-accumulator dependency chain that limits
   the naive version. Factorizing is faster than not factorizing, which is a
   pleasant inversion of the usual accuracy/speed tradeoff.
3. **`dp4a` is sufficient — tensor cores are not needed for memory-bound decode.**
   At 94% of DRAM roofline the MACs are idle regardless. Save IMMA
   (`m16n8k32` int8, or `m16n8k64` u4 — Q2_K values fit in 4 bits) for the
   **prefill/large-M** regime where the kernel actually becomes compute-bound.

**Honest scope of the measurement:** this probe exercises the weight-stream inner
loop with representative unpack+reduce arithmetic — the dominant cost — but not
smem staging, the cross-K reduction, or output writes. Expect **80–88%
end-to-end**, i.e. **~2.7× on `w13` and ~7× on `w2`** versus today.

What the probe establishes is **the ceiling is reachable and the unpack is free**.
It does *not* establish that the integer/`dp4a` route is the best way to get there
— §2.4 argues it is not.

### 2.3 Why the current kernels are at 12–32%: the specific mechanisms

Reading the GGUF Q2_K path against Marlin makes the gap concrete, and it is
structural rather than a matter of op count:

| | GGUF Q2_K | Marlin |
|---|---|---|
| global→shared | **synchronous, register-staged** | 4-stage `cp.async.cg` ring |
| `cp.async` anywhere in `csrc/quantization/gguf/` | **zero occurrences** | throughout |
| math unit | **`dp4a` only** (CUDA cores) | `mma.m16n8k16` / `m16n8k32` tensor cores |
| double buffering | none; 2 `__syncthreads` per iteration with nothing in flight | register `[k%2]` + 4-stage shared ring |
| output tile width | `MMQ_X_Q2_K = 4` → **the weight tile is re-read from HBM every 4 tokens** | 16–64 |
| load balance | 2D grid, whatever tail falls out | one-wave striped stream-K + L2-resident serial reduce |
| scale decode | `m \|= m<<8; m \|= m<<16` and `sc & 0xF` re-executed **per output element per k-block** | decoded once per group into `frag_s`, reused |

Two items dominate. First, **`MMQ_X = 4`**: at any batch above 4 the same expert
weights are streamed from DRAM repeatedly, which alone explains why `mmq` time is
flat in batch while sitting at 0.2–9% of peak. Second, **8 of every 16 `dp4a` are
waste** — they compute `m_j·Σx` against a broadcast constant, recomputed for every
output row and every expert with no amortization. For a 4096×4096 decode step
that is ~4M redundant `dp4a` per token per layer.

Marlin's ~1.6 ops/weight are amortized across the whole mma tile (one dequant
feeds `thread_m_blocks` mma instructions); GGUF's ~1.5 ops/weight are paid **per
output element**. Same op count, entirely different throughput.

### 2.4 The better primary design: fold the min into the fp16 fragment

This supersedes the integer-epilogue recommendation above. Marlin's
`scale_and_sub` applies scale *and* offset in a single instruction:

```
scale():          frag_b = __hmul2(frag_b, s2)                 // 2 ops / 4 weights
scale_and_sub():  frag_b = __hfma2(frag_b, s2, __hneg2(zp2))   // 2 ops / 4 weights
```

**Identical instruction count.** Setting `s2 = d·sc_j` and `zp2 = dmin·m_j` makes
Q2_K's asymmetric two-parameter dequant cost **exactly the same as symmetric
int4** — and the `zp2` pre-multiply (`frag_zp = __hmul2(frag_zp, frag_s)`) is
hoisted once per group. So:

> **The `Σx` factorization is strictly dominated on a tensor-core kernel.** The min
> term is free in the fp16 operand, with no 16-granular sum buffer, no extra
> rank-1 term, no epilogue changes, and no activation quantization error.

The full budget on this route: **0.56 ops/weight** for the unpack (8 `lop3` +
1 shift per 16 weights, using `0x64006400` = fp16 1024.0 and absorbing the
powers of two — `{1024+v, 1024+4v, 1024+16v, 1024+64v}` — into the per-16 scale,
so **no shifts appear in the inner masks**) plus **0.5 ops/weight** for
`scale_and_sub` ≈ **1.06 ops/weight against the 1.83 budget**. DRAM-bound with
margin, *and* it runs on tensor cores so the same kernel serves prefill.

Use **fp16, not bf16**: int8→bf16 costs 2.5 ops/weight (`byte_perm` + fp32
subtract + repack, and the scale cannot be fused) versus 1.0 for fp16. Accumulate
in fp32 regardless.

Keep the integer/`dp4a` + `Σx` route as the **int8-activation variant only**,
where a fractional min genuinely cannot be folded into an int8 operand. There,
extend `quantize_q8_1` to emit 16-granular sums and contract `m_j` as a separate
rank-1 epilogue term.

### 2.5 Kernel structure

- **Layout repack at load time, layout-only.** Permute Q2_K blocks so
  shared→register loads land directly in mma-B-fragment order, eliminating every
  runtime shift and shuffle — Marlin's highest-leverage trick, and the one that
  most directly attacks Q2_K's native layout, where a sub-block's 2-bit values are
  spread across four bit-planes of the same bytes (`(x_ql[k+l] >> shift) &
  0x03030303` with a non-linear `shift` expression). Follow the repack recipe:
  gather with the mma B-operand k-offsets `{0,1,8,9}`, pack with a bit-position
  permutation analogous to `pack_idx = {0,2,4,6,1,3,5,7}`, and store thread-major
  so the GEMM's `cp_async4` is perfectly coalesced and the shared read needs **no
  XOR swizzle**. Target 32-bit words holding 16 values = 8 k × 2 n-columns.
  Two constraints, both free at runtime: each `lop3` output's two lanes must
  belong to the *same* 16-element scale group, and they must land in mma register
  order.
- **Do not expand the scales to flat fp16** — that grows 84 B → 128 B/256
  (2.625 → 4 bpw, **+52% bytes**), and on a DRAM-bound kernel that is a direct
  ~35% throughput loss to save ALU work we have shown is affordable. Instead
  **decode narrow scales in-register**: store the 4-bit `(sc_j, m_j)` pairs and
  the per-superblock `(d, dmin)`, and unpack in `fetch_scales_to_registers` — the
  same hook the NVFP4/MXFP4 paths already use for narrow-format scale decode.
  Cost ≈ **0.16 ops/weight** (one 32-bit scale word covers 4 sub-blocks = 64
  k-elements). Bytes stay at 2.625 bpw.
- **cp.async ring, 4 stages**, `.cg` (bypasses L1, no extra registers), with
  `wait_group<stages-2>` and the global issue placed **two k-steps before the end
  of the k-loop** so the final dequant+mma work hides the barrier. Fence
  unconditionally even when winding down, so the group count never desynchronizes.
  **Do not add mbarrier split arrive/wait**: Marlin deliberately omits it on
  sm80 — it costs a shared-memory atomic per stage and buys nothing without TMA
  or warp specialization. (This corrects an earlier recommendation here.)
- **163 KB smem/block** (vs 99 KB on sm86) allows a deeper ring than the 3090-era
  code assumes. Requires explicit opt-in via
  `cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, …)` plus a
  164 KB carveout; static smem stays capped at 48 KB.
- **`ldmatrix.x4`** for activations (which cannot be pre-permuted, so they keep the
  XOR-swizzled shared layout), and **`.x2` with swapped mma operands for M ≤ 8** —
  that halves the mma count at decode batch sizes.
- **Occupancy target 4 blocks/SM.** Measured sweet spot; 2 is latency-starved and
  8 costs bandwidth (1521 GB/s [M]). Also adopt the occupancy rescue rule: if
  `prob_n/thread_n × ceil(M/tb_m) × 4 ≤ 108`, narrow `thread_n` to 64 — at decode
  batch sizes this triggers constantly.
- **Work partitioning:** one-wave grid with the two-phase split — whole `(m,n)`
  tiles that own their full K reduction (no cross-block reduce) for the bulk, then
  stream-K over the ragged remainder with an L2-resident serialized reduce
  (`ld.global.acquire` spin / `red.relaxed.global.add` release, accumulating in
  place in the output buffer). **Align the stripe length to superblock
  boundaries**: at `thread_k = 128` a 256-element superblock spans 2 k-tiles, so
  round the per-block iteration count to a multiple of 2 or handle mid-superblock
  starts explicitly.
- **Delete the act-order path entirely.** Q2_K has no activation reordering, so all
  the `g_idx` staging, `init_same_group`, heterogeneous `scale4` assembly, and the
  column-permute kernel can go — worth ~16 registers and a chunk of shared memory.

### 2.5b Implementation status and what the first landing actually measured [M]

`kernels/quant/q2k_ampere.cuh` + `q2k_ampere_test.cu` implement the integer route.
Correctness is solid: the repack round-trips **bit-exact** against the native
q2_K formula (135168 spot checks), and the GEMV matches an fp64 host reference to
**0.61% relative** -- the expected int8-activation error floor (~1/127).

Throughput, N=32768 K=6144 (63 MiB of weights, well past the 40 MB L2):

| M | GB/s | Gweight/s | % of 1769 ceiling |
|--:|--:|--:|--:|
| 1 | 775 | 2361 | **44%** |
| 2 | 616 | 1877 | 35% |
| 4 | 350 | 1065 | 20% |
| 8 | 159 | 484 | 9% |

(Config `NR=4, QB=2, KC=4096`. Reproducible to <1% across GPUs; a standalone
sweep harness reads 10-15% higher for the same config, so treat cross-harness
comparisons with the usual caution and compare within one harness.)

**Time scales linearly with M, so this kernel is compute-bound for M >= 2** --
batching buys no bandwidth amortization. That is a real correction to this
document's earlier expectation of 80-88% end-to-end, and it relocates one roadmap
item:

- At M = 1 the kernel is genuinely near the useful limit (47% of the raw stream
  ceiling; the probe's 94% had its activation in a register with no loads, no
  scale traffic and no row-strided access, so it was always an upper bound).
- For M >= 2 the binding constraint is **ops per weight byte**: ~13.5 against the
  5.57 budget, dominated by the per-(row, sub-block, m) float epilogue
  (`acc += g*(dsc*A - dmm*S)`, 6 float ops) and 4 `dp4a` per m.

Two fixes, in order, both now higher priority than originally scheduled:

1. **Move the M dimension onto the tensor cores.** `dp4a` retires 4 MACs per
   instruction; IMMA `m16n8k32` retires 512. The M loop is currently scalar, which
   is why cost scales with M. **The IMMA crossover is M ~= 2, not "prefill"** --
   the roadmap previously deferred IMMA to large-M only, which these numbers show
   is wrong.
2. **Coarsen the activation scale group from 32 to 256** so `g` is constant across
   a superblock. Then `SUM_j sc_j*A_j` and `SUM_j m_j*S_j` accumulate in int32 for
   a whole superblock and the float epilogue drops from 6 ops per (row, sub-block,
   m) to ~4 per (row, superblock, m) -- a ~10x reduction in the epilogue term.
   Activations are the more precise side (int8 vs 2-bit weights), so the accuracy
   cost should be small, but it must be measured.

Steps already applied, with their measured effect, so they are not re-litigated:

| change | effect at M=4 | note |
|---|--:|---|
| n-tiling (NR rows/warp) so activations are not re-read per output row | 7% -> 13% | |
| stage activations in smem once per block; sequential weight walk | 13% -> 16% | |
| independent weight loads in flight (the cp.async substitute) | 16% -> 22% | |
| split activations into 4 conflict-free smem planes | no change | conflict was real but never binding |
| per-superblock activation scale + QB=2 sub-block batching + retuned NR | 20% | **+40% at M=2**, ~neutral elsewhere |

Two lessons worth keeping. First, the 4-way bank conflict was real and removing
it changed nothing -- measure before optimizing. Second, **register pressure
dominates tile-size choice here**: `NR=8, QB=4` looks strictly better on an
ops-per-byte count but costs 127 registers, drops to 2 blocks/SM, and loses 37%
at M=2 versus `NR=4, QB=2`. The measured (NR, QB) sweep:

| config | M=1 | M=2 | M=4 | M=8 |
|---|--:|--:|--:|--:|
| NR=8 QB=4 | 800 | 473 | 256 | 130 |
| **NR=4 QB=2** | 877 | **747** | **428** | **195** |
| NR=2 QB=2 | **1081** | 738 | 419 | 187 |

Use `NR=2` for a pure M=1 workload (1081 GB/s = 61% of the stream ceiling).

### 2.6 `w13` and `w2` need different tilings

| tensor | shape | K | superblocks per output |
|---|---|--:|--:|
| `w13` (fused gate+up) | [256, 2048, 6144] | 6144 | 24 |
| `w2` (down) | [256, 6144, 1024] | 1024 | **4** |

`w2`'s 12% (vs `w13`'s 32%) is a **short-K** problem: only 4 superblocks per
output element, so per-output overhead (index math, scale loads, output write,
reduction setup) is amortized over 6× less work. It needs more output rows per
block, not more K-splitting. Treat them as two separately tuned kernels; do not
let one tile config serve both.

### 2.7 Q8_0 (the dense/attention half: 872 tensors)

Q8_0 stores int8 directly, with one fp16 scale per 32 — and it is **almost exactly
Marlin's existing `kU8B128` format**: add 128 at repack time to convert
signed→unsigned-with-bias, set `group_size = 32`, and the instantiation already
exists. The dequant is `prmt`-based (interleave a byte of `q` with the constant
`0x64` to build fp16 `1024 + byte` in one instruction) plus one `__hsub2`:
**1.0 op/weight**, or 0.5 with the bias folded into the scale.

Use **fp16 compute here too**. The int8→bf16 route costs 2.5 ops/weight
(`byte_perm` + fp32 subtract + repack) and the scale cannot be fused into it —
2.5× worse for no benefit.

With a 5.92 ops/weight budget this should reach 85–90% of roofline versus the
measured 53–60%. **This is the cheapest large win in the whole document**: it
touches 872 tensors (vs Q2_K's 228), and most of the kernel already exists
upstream.

---

## 3. The second constraint: launch overhead

Measured on this host [M], 500 tiny kernels (≈ the node count of one decode step
across ~75 MoE layers × 2 GEMMs plus attention):

| dispatch | per launch | 500 nodes | implied cap |
|---|--:|--:|--:|
| streamed | 3.01 µs | 1.50 ms | 667 tok/s |
| **CUDA graph** | **1.05 µs** | **0.53 ms** | **1896 tok/s** |

Against the weight-bound MoE cost of one decode token (8 of 256 experts,
75 MoE layers, 49.5 MB/layer → 3.71 GB/token):

| config | MoE bandwidth time | graph launch time | **bound by** |
|---|--:|--:|---|
| 1 GPU | 2.23 ms | 0.53 ms | bandwidth |
| TP8 / EP8 | **0.28 ms** | **0.53 ms** | **launch overhead** |

**At TP8 on A100, a graph-resident decode step is launch-bound, not
bandwidth-bound.** Reducing node count therefore outranks kernel tuning in the
low-batch regime — and CUDA graphs are mandatory, not optional (2.9×).

### Two regimes, two strategies

- **Batch 1–8 — latency/launch-bound.** Fuse aggressively: gate+up+down into one
  MoE launch, norm+int8-quant into the GEMM prologue, router into the dispatch.
  Prefer a **persistent** MoE kernel over per-expert launches. Speculative
  decoding (DSpark is already in-tree) divides launch cost by the acceptance
  count — worth more here than any kernel change.
- **Batch 16–128 — DRAM-bound.** The §2 kernel is the whole game.

---

## 4. Ampere mechanisms this design depends on

From the SM 8.0 feature set, ranked by value to this workload:

| mechanism | use here |
|---|---|
| 163 KB smem/block (vs 99 on sm86) | deep cp.async rings; **needs explicit opt-in** |
| `cp.async` (bypasses L1, no extra registers) | weight streaming; frees registers for accumulators |
| `ldmatrix.x4` / `.x2` + swapped mma operands at M ≤ 8 | activation fragment loads; halves mma count at decode batch |
| `__reduce_add_sync` int32 (**sm80-only**) | cross-lane reduce in the int8-activation variant only |
| ~~mbarrier split arrive/wait~~ | **do not use** — costs a shared-mem atomic per stage and buys nothing without TMA/warp specialization on sm80 |
| 32 blocks/SM, 64 warps/SM (vs 16/48 on sm86) | more resident concurrency for latency hiding |
| IMMA `m16n8k32` s8 / `m16n8k64` u4 | **prefill only** — decode is DRAM-bound |
| 40 MB L2 + `cudaAccessPolicyWindow` persistence | pin router weights, indexer hot pages, KV metadata — **not** expert weights (1.585 GB/layer, no reuse) |
| NVLink3 600 GB/s bidirectional | TP all-reduce; keep remote windows **under the 64 GB Link TLB reach** |

Build note: compile explicitly for `sm_80`. An `sm_86` binary will not launch
here at all, and an `sm_80` binary forfeits sm86's 2× FP32 — the two Ampere tiers
need separate targets.

---

## 5. Parallelism and capacity

- 244 GiB Q2_K on 80 GB cards → **TP8 minimum** (~30.5 GiB/GPU of weights, plus
  KV). IQ2_XXS at 196 GiB makes TP4 feasible and is the better first target.
- **TP vs EP for the MoE:** at batch 1, EP8 leaves each GPU holding 32 experts
  with only ~1 active → 6.2 MB of work → ~4 µs, i.e. pure launch overhead. TP8
  splits every expert so all GPUs work every token, trading an all-reduce for
  latency. **Recommend TP for the MoE at low batch, EP at high batch**, switched
  by the same threshold that selects the two regimes in §3.
- The MLA latent does not shard under TP, so extra GPUs buy batch, not context.

---

## 6. Prioritized roadmap

| # | work | regime | expected | confidence |
|---|---|---|--:|---|
| 1 | **Q8_0 dense/attention GEMM** — `kU8B128` relabel (+128 at repack, group 32), fp16 compute. Most of the kernel exists upstream. | all | 53–60% → **85–90%** | high — 872 tensors, least new code |
| 2 | Load-time **layout-only repack** for Q2_K and Q8_0 (mma-fragment order, no byte growth) | all | prerequisite for 1 and 3 | high |
| 3 | **Q2_K IMMA path** — move the M dimension onto `m16n8k32` int8 tensor cores; the landed dp4a kernel is compute-bound from M≥2 (§2.5b) | batch ≥2 | unblocks batching | high — dp4a retires 4 MACs/instr vs 512 |
| 3b | Coarsen activation scale group 32→256 so the epilogue accumulates in int32 per superblock (§2.5b) | all | ~10× on the epilogue term | medium — needs an accuracy check |
| 4 | Node-count reduction + mandatory CUDA graphs (fuse MoE, norm+quant, router) | batch 1–8 | up to **2.9×** on the launch term | high — 1.05 vs 3.01 µs measured |
| 5 | Sparse-MLA decode: gather-dequant to bf16 scratch, then dense MQA | all | bounded by topk, not context | medium |
| 6 | Indexer logits kernel (replaces the pure-torch stub) | long context | removes the stub cliff | medium |
| 7 | Stream-K / MoE grouped tile enumeration for the ragged expert distribution | batch ≥16 | load balance | medium |

**What not to do:**

- **Do not expand Q2_K scales to flat fp16.** +52% bytes on a DRAM-bound kernel;
  decode the narrow scales in-register instead (§2.5).
- **Do not build the `Σx` factorization for the fp16 path.** `scale_and_sub` folds
  the min in for free; the factorization only pays in an int8-activation variant
  (§2.4). *This reverses an earlier recommendation in this document.*
- **Do not add mbarrier** on sm80 (§4).
- **Do not use bf16 for the quantized GEMMs** — 2.5× the dequant ops of fp16, and
  the scale cannot be fused.
- **Do not carry over any 3090-derived threshold** (`qgemm_ksplit`'s
  `tiles < 832`, the mmvq/mmq crossovers) without re-deriving — the
  compute:bandwidth ratio moved by 4×.
- **Do not keep `MMQ_X = 4`.** Re-reading the weight tile every 4 tokens is the
  single largest structural defect in the current path (§2.3).

---

## 7. Open questions

- **The DSA indexer cadence is the #1 long-context question, and it is
  checkpoint-determined.** `deepseek_v2.py:1092-1101` gates each layer's indexer on
  `index_topk_freq` (**default 1 — i.e. every layer**) or an explicit
  `index_topk_pattern` string where `"S"` marks a skipped layer; skipped layers
  build no indexer and reuse the previous layer's indices. The arithmetic at
  `freq=1`, 100k context, with the indexer's own fp8 K cache
  (head_dim 128 + 4 B scale = 132 B/token/layer):

  | | per token |
  |---|--:|
  | indexer scan, 78 layers × 13.2 MB | **1.03 GB → 590 µs** at roofline |
  | MoE weights at TP8 (§3) | 0.28 ms |

  So at `freq=1` the indexer scan **exceeds all other traffic combined** and is
  the binding constraint at long context — consistent with the fork's own
  observation that prefill falls from ~800 to 541 tok/s at 1M while decode stays
  flat (decode attention is bounded by `index_topk=2048`; the *scan* is not
  bounded by anything). **Read the actual `index_topk_freq`/`index_topk_pattern`
  from the checkpoint before optimizing anything else for long context** — it
  changes this term by the cadence factor and nothing else in this document by
  more than a few percent.
- End-to-end efficiency of the §2 structure once smem staging, K-reduction and
  output writes are included (probe says 94% for the inner loop; 80–88% expected).
- Accuracy of int8 activations against a Q2_K weight set. GGUF's `q8_1` path is
  the established reference so this should be a non-issue, but it needs a
  measured eval, not an assumption.
