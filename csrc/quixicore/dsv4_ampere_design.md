# DeepSeek V4 0731 on Ampere A100

## Goal

DeepSeek V4 Flash 0731 must run its routed experts natively from the GGUF
quantized weights on A100:

```text
IQ2_XXS gate/up -> SwiGLU -> Q2_K down -> weighted reduce
```

The implementation is owned by SlimServe, supports the combined vLLM
`[expert, gate | up, packed]` W1 layout, and must not dequantize or retain an
expanded copy of the expert weights. TP2 and TP4 use the same all-layer
IQ2_XXS/Q2_K model when measuring scaling.

## Reference Lessons

The optimized ROCm path establishes that IQ2_XXS must decode its data-dependent
codebook at tile load, once per weight, then use a signed-byte dot-product
inner loop. Decoding inside every row/column dot discards the benefit of a
tiled kernel. Its 8-route tile also establishes that codebook decode needs
enough activation reuse to amortize it.

The GLM 5.2 A100 work establishes the Ampere rules: keep packed weights
byte-neutral, separate gate/up and down tilings, use `cp.async` and tensor
cores when routed M is large enough, keep low-batch decode launch count small,
and treat tile width as a kernel property rather than a routing-layout
property.

DS4 establishes the useful operation boundary: pair gate and up, reuse the
input quantization and routing schedule, feed Q2_K down directly from a
quantized SwiGLU intermediate, and perform router weighting only after down.

## Implemented Stage 1

`quant/dsv4_moe_ampere.cuh` implements the first production path:

1. Quantize each input token to Q8_1 once.
2. Group routed rows by expert using vLLM's aligned schedule.
3. Load paired gate/up IQ2_XXS tiles from the combined W1 tensor.
4. Decode each IQ2_XXS codebook entry once at tile load.
5. Reuse one staged activation tile for both gate and up dot products.
6. Apply SwiGLU in registers and emit Q8_1 directly.
7. Run native grouped Q2_K down and final weighted reduction.

This removes the fp32 gate/up tensor, fp32 SwiGLU tensor, one activation
quantization launch, and duplicate gate/up activation staging. It does not
expand or repack the resident model weights.

W1 dispatch is 4 routes below 256 routed rows and 8 routes at or above 256.
W2 remains 4 routes. The Python scheduler aligns to the least common multiple
and expands the expert-id view independently for W2, so changing W1 width
cannot silently change W2's expert selection.

## Measured Stage 1

Synthetic valid GGUF blocks at the TP2-local model shape (`H=4096`, local
`I=1024`, top-8) pass finite-output and generic-path parity checks. The first
recorded comparison is in `perf/optimization_status.md`; the checked-in harness
is `benchmarks/kernels/benchmark_dsv4_moe_a100.py`.

Stage 1 was not the final kernel design. The retained decode path described
below supersedes its routed down-output materialization.

## Implemented Stage 2: Short-K Q2_K Down

DSV4 down is a different problem from gate/up: under TP2 its local K is 1024,
only four Q2_K superblocks, while N is 4096. The retained kernel therefore
favors more output rows per CTA, hoists narrow `(scale, min)` decode, and
avoids a generic long-K schedule.

Two writeback strategies were measured:

- Grouped expert output followed by the current deterministic weighted reduce.
- A token/output-tile kernel that owns all top-k slots and reduces router
  weights in the CTA, avoiding the routed down tensor without global atomics.

The retained `q2_k_down_sum_repacked` kernel owns a token/output tile across
all selected experts and writes the final BF16 local TP partial directly. Its
byte-neutral per-expert Q2_K layout separates packed quants, scale/min nibbles,
and `d/dmin` pairs without storing dequantized weights. At TP4-local `K=512`,
eight warps with two output rows per warp is the measured geometry; TP2 uses
the corresponding `K=1024` specialization. Atomic accumulation and standalone
Q2_K GEMV are not production paths.

## Stage 3: Ampere Tensor-Core Layout

For routed M >= 2, implement a load-time, byte-neutral permutation for IQ2_XXS
and Q2_K into MMA-fragment order. The resident format keeps 2-bit values and
narrow scales packed; decode occurs in registers/shared memory, never as a
full dequantized tensor.

- Use a 4-stage `cp.async.cg` ring for weight tiles.
- Decode IQ2 codebook values once into signed-byte shared tiles.
- Use `mma.sync.aligned.m16n8k32` for batched signed-byte products where it
  beats dp4a.
- Decode Q2_K scale/min pairs in registers and fold them into the fragment
  epilogue; do not expand scales to fp16 storage.
- Keep a low-M dp4a path when the tensor-core setup cost loses in measurement.
- Replace raw GGUF tensors at load time or load split/aligned artifacts; never
  keep raw and repacked expert stacks simultaneously on TP2.

The current decode path has implemented the byte-neutral IQ2_XXS and Q2_K
load-time layouts and low-M dp4a specialization. Tensor-core MMA remains a
measured crossover option for routed `M >= 2`; it is not assumed to beat dp4a
for concurrency-one decode.

## Stage 4: Distributed Ownership And Producer-Progress Collectives

The retained TP4 graph proves that faster isolated GEMV or mHC kernels do not
solve scaling. The exact TP2/TP4 fit leaves a 7.137 ms non-scaling floor and
TP4 needs to recover 1.734 ms/token. There are 85 foreground mHC transitions
per token. The routed transition is currently two graph nodes:

```text
q2_k_down_sum_repacked -> all_reduce_dsv4_mhc_add
```

The attention transition has the analogous boundary:

```text
aligned_q8_0_q8_1_gemv -> all_reduce_dsv4_mhc
```

Node elimination alone is not the target. The materialized producer ends only
0.3-0.9 us before its local urgent launch, while rank arrival skew is 10-16 us.
A useful fused boundary must publish producer tiles in a deterministic order
and reduce ready peer tiles while the producer tail is still running. Waiting
for a whole cooperative producer grid and then executing the existing mHC
handshake preserves the expensive arrival barrier and has already measured
neutral.

Start with routed MoE: carry the repacked W2 pointer, Q8_1 SwiGLU intermediate,
top-k ids, local shared-expert addend, and mHC state to a custom-allreduce-owned
decode op. Use a fixed persistent CTA set on every rank, per-tile readiness
counters, and a deadlock-safe identical tile traversal. Map reduced hidden
tiles directly into the established 16 logical mHC partials. Preserve
rank-ordered BF16 reduction and the existing deferred-stream event contract.
The aligned-Q8 attention boundary needs the same progress protocol; its current
thousands-of-CTA row grid cannot safely block waiting for peer CTAs that may
not have been scheduled.

The Python ownership change is explicit: the GGUF MoE path must return a
short-lived pending native-producer descriptor during decode instead of
launching Q2_K eagerly. `DeepseekV4DecoderLayer` consumes that descriptor only
at the immediately following mHC boundary. Warmup, eager fallback, prefill,
and unsupported shapes continue to materialize the ordinary tensor. Pending
state must be per-forward-call and graph-capture safe; a process-global
last-operation singleton is not acceptable.

Do not implement this as Q2_K followed by a second fused mHC launch, and do not
split deferred finalization into another graph node. Both forms have already
lost end to end. The acceptance test is exact TP4 throughput against a
same-session materialized control, followed by the canonical TP2/TP4 sweep.

This collective work is only one part of the scaling correction. The 64-head
Lightning indexer is explicitly replicated today. Its production replacement
must shard query heads and head weights, compute local partial logits, and make
the existing persistent top-k node reduce peer partial scores before selection.
The rejected token-sharded implementation performed a generic all-gather and
is not the design to repeat. The two sparse-MLA source launches should also be
merged into one source-selecting persistent kernel, preserving their current
per-source partial slots and final reduction tree.

## End-to-End Gates

Every retained stage must pass the exact-token SlimServe benchmark with the
same model and workload at TP2 and TP4. TP4 must be at least 1.5x TP2; failure
means the design, collectives, graph shape, or scheduler remains wrong. Kernel
timings, exact throughput, correctness, and raw logs belong in `perf/`.
