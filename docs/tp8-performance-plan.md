# Getting maximum performance from TP8 on Kimi K3

## The governing constraint

K3 decode is **overhead-bound**, not work-bound. Four independent measurements
on this box say the same thing:

| measurement | result | reading |
|---|---|---|
| whole decode step vs HBM roofline | 4.4% | not bandwidth |
| MLA decode kernel | 55–280 GB/s of 5300 | not bandwidth |
| one 56 KiB collective | 54 µs | ~98% latency |
| MoE work halved (512 → 256 units/rank) | 3–7% faster | not work |
| MoE tokens deduplicated 8× | **slower** | not work |

The last two are decisive. Time is set by how many kernels and collectives run
in sequence, not by what is inside them. Kernel count per rank does not change
with `tp_size` — every rank still walks 93 layers — which is why adding two GPUs
has so far produced nothing.

## Why TP8 can still win

Because the overhead is removable, and once it is gone the step becomes
work-bound — and *then* 8 ranks beat 6. The order matters: distributing work
more finely first (finer MoE splits, better all-to-all) is measurably worthless
while the step is overhead-dominated.

Current: `k3-6` 34.0 / 120.4 tok/s, `k3-8` 31.1 / 94.8 (c1 / c8).
TPOT: 28.74 / 62.85 ms vs 31.45 / 80.98 ms.

## Phase 0 — attribute the missing 13 ms (do this first)

18 ms of the c8 TPOT gap is unexplained. Collectives are equal (EP8 10.08 ms vs
TP6 10.63 ms) and MoE row count is disproven. Do not build anything until this
is attributed.

`rocprofv3` aborts on tool registration here and TP8's workers are separate
processes, so use in-graph microbenchmarks per op class, the way MLA (4.7%),
KDA (1.0–1.5%) and `mul_mat_vec` (~17%) were attributed. Sum them and find the
residual.

Exit gate: every millisecond of the 80.98 ms accounted for, or the largest
single unattributed op class named.

## Phase 1 — delete expert parallelism from TP8

EP adds a dispatch, a combine and expert-indexing kernels to every one of the 93
MoE layers, on top of an unchanged sequential depth. That is the one structural
difference from TP6, and it is pure overhead on this workload.

Blocker: tensor-parallel MoE is illegal at 8 ranks. `w2` is row-parallel, so TP
splits its packed **byte** axis: 1008 bytes of Q2_K over 8 gives 126, and
`type_size` is 84, so every rank starts decoding 1.5 blocks in. Proven by
tracing a live load.

Two legal shardings, both block-aligned:

- **Pad** the intermediate 3072 → 4096. Uniform 512 units (2 blocks) per rank,
  ranks 0–5 holding the 12 real blocks and 6–7 only zeros. ~33% more MoE weight
  memory; two ranks idle in the MoE.
- **Uneven**, 4 ranks × 2 blocks + 4 ranks × 1 block. No waste, all ranks busy,
  but `intermediate_size_per_partition` becomes rank-dependent.

Prefer padding first: it is smaller, keeps every shape uniform, and the point is
to measure whether removing EP's kernels helps at all. The per-rank MoE work is
then identical to TP6 (512 units), which the scaling measurement says is fine —
work is not what costs.

Implementation sites: `_materialize_gguf_moe_param` (divides the byte axis by
`tp_size` with no block awareness), plus a clamp where `rank * shard_bytes >=
src_bytes` so the tail ranks load nothing and keep their zeros.

Expected: removes ~93 collectives and the per-layer indexing kernels.
Exit gate: TP8 within noise of TP6, coherence-gated. If it is not, the residual
from Phase 0 is the real problem and Phase 2 is where the win lives.

## Phase 2 — cut kernels per layer (the actual lever)

This helps TP6 and TP8 alike, but it is what makes TP8's extra hardware
matter, because it moves the step toward work-bound.

Ranked by measured cost:

1. **The GGUF MoE kernel is overhead-bound.** 76 µs at 8 tokens for work that is
   microseconds; halving the work changes it 7%. Two calls per layer (w13, w2)
   × 93 layers ≈ 14 ms of the 81 ms step. Fuse w13 and w2 into one launch, or
   fuse the routing/index/gather prologue into the matmul.
2. **`mul_mat_vec` is ~17% of the step** at 5–9 µs per call, hundreds of calls.
   Its `quantize_row_q8_1` prologue is a separate launch per matmul — fuse it.
3. **93 layers × 2 collectives.** Both TP and EP pay ~10 ms. A HIP peer-to-peer
   all-reduce over XGMI (the pattern `custom_all_reduce.cuh` already uses) can
   beat RCCL's 54 µs on 56–112 KiB messages, where ~98% is latency. Note this
   was measured *not* to be the TP8-vs-TP6 differentiator — it is a win for both.

Exit gate: step time down measurably with coherence intact, after each item.

## Phase 3 — then, and only then, TP8's hardware advantage appears

Once the step is work-bound rather than launch-bound, 8 ranks carry 33% more
bandwidth and compute than 6, and TP8's 12 attention heads per rank are 25% less
work than TP6's 16. Re-measure the TP6/TP8 comparison after Phase 2; the
ordering should invert.

## What not to do

- **Do not write a HIP all-to-all to beat MORI.** Measured: EP8 collectives
  10.08 ms vs TP6 10.63 ms. Not the differentiator.
- **Do not enable `use_sequence_parallel_moe` for TP8/dp1.** Tried: TPOT 31.45
  → 82.15 ms at c1. The `dp > 1` gate is doing real work.
- **Do not repack `w2` transposed to get a finer MoE split.** Measured: 384 vs
  512 units per rank is a 3–7% difference. Not worth a requantization.
- **Do not trust `--ignore-eos` throughput alone.** It reported 154 tok/s on a
  model emitting `!!!!!!!!`. The harness now gates on a known answer; keep it.
