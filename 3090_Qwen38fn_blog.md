# 1,200 tokens/second on eight RTX 3090s: serving Qwen3.8-Flash-Next at full 262K context, with a half-terabyte KV cache in system RAM

**Eric Hartford, QuixiAI - August 2026**

Qwen3.8-Flash-Next is a 125B-parameter vision-language model with 6B active
parameters per token - a hybrid of Gated DeltaNet linear attention and Qwen
Sparse Attention, with only one full-attention layer in four, a four-branch
gated residual stream, 51 GiB of n-gram embedding tables, and a built-in
single-layer MTP drafter. It is a genuinely strange architecture, and it is
exactly the kind of model that consumer hardware was never supposed to serve.

We serve it on eight RTX 3090s - five-year-old, 24 GB GeForce cards with no
FP8 tensor cores and no NVLink - at its **native 262,144-token context**,
with **exact bf16 KV cache**, and a **512 GiB pinned-RAM second tier** that
lets conversations evicted from GPU memory resume in under a second instead
of re-prefilling minutes of history.

The numbers, measured with an exact-token harness (1,000 tokens in / 2,000
out per request, the model's shipped sampling defaults, seeded):

| Concurrency | Aggregate tok/s | Per-request tok/s | Median latency |
| ---: | ---: | ---: | ---: |
| 1 | ~140 (129.8-157.8) | ~140 | 14 s |
| 8 | 590-600 | 81 | 25 s |
| 32 | **1,091-1,199.8** | ~40 | 50 s |

Peak recorded: **1,199.8 tok/s at 32 concurrent requests**. Our first
working configuration managed 409.7 tok/s at half the context - the campaign
bought a net **2.9x** while doubling context to the model's native maximum
and *removing* KV quantization. Here is how, trick by trick.

---

## SlimServe: opinionated serving, or nothing

Everything in this post ships as part of **SlimServe**, our simplicity-first
inference engine - vLLM's engine underneath, [ds4](https://github.com/antirez/ds4)'s
interface philosophy on top, and every bleeding-edge performance trick we
can land in between.

The idea behind SlimServe is a deliberate inversion of how serving stacks
usually work. General-purpose engines expose hundreds of flags and let you
discover, eight hours into a 244 GiB download, that your combination
doesn't fit, silently disables the feature you needed, or serves at a tenth
of the hardware's potential. SlimServe refuses to be general-purpose:
**every legal configuration is a profile** - model x quant x platform x
config - recorded in one registry file with every setting annotated by the
measurement that put it there. You type `slimserve qwen38fn-fp8-8 --serve`;
it detects the machine, downloads and checksums exactly the blessed
artifacts, and runs the one configuration that was tuned and validated on
that hardware. Anything else is refused with a readable reason. Support for
models and platforms outside the tested set has been *deleted*, so the rest
can be tuned hard - this whole campaign is what "tuned hard" means for one
profile.

The policies in this post - prefix caching, tool calling, and thinking
always on; never greedy; exact bf16 main KV - aren't documentation, they
are registry-level defaults enforced by tests. A profile cannot silently
regress them.

Underneath the profiles sits **QuixiCore**, our kernel library and the
foundation of everything SlimServe does on-device. QuixiCore exists because
the models we serve and the hardware we serve them on are both outside the
mainstream's tuning envelope: nobody upstream is optimizing MFMA quantized
GEMMs for a 244 GiB routed-MoE GGUF on MI300X, or packed sparse-MLA decode
caches and fused gated-DeltaNet steps on Apple Metal, or - as in this post -
sparse-attention gather tiles on FP8-less Ampere GeForce cards. So we write
them: [QuixiCore-rocm](https://github.com/QuixiAI/QuixiCore-rocm) carries
the gfx942 HIP/MFMA kernels that make GLM-5.2-Vision serve on two GPUs at
all; QuixiCore-Metal is vendored whole into SlimServe as the native MPS
serving path for three model families; and the CUDA line grows the same
way this campaign did - kernels are developed *in the serving repo first*,
against the real profile and the real workload, then the finished pieces
are ported back into the library. Vendored kernel code lives under
`csrc/quixicore/`, and the rule is strict: only kernels on an actual
serving path get vendored, and every one is tuned against measurements from
the profile that uses it. The QSA sparse-gather tile pinning in trick #7
below and the Marlin-lineage quantized-GEMM work on Ampere are this
process in action.

QuixiCore is also where we go to break marketing narratives. NVIDIA's
positioning says NVFP4 is a Blackwell feature - you want the new 4-bit
format, you buy the new silicon. With QuixiCore kernels, **NVFP4
checkpoints serve on any hardware**: SlimServe runs Qwen3.8's NVFP4 build
on AMD MI300X - gfx942 has no FP4 hardware at all; the packed E2M1 weights
decode in-register through a dp4a-style quantized GEMV, measured 1.3-1.8x
faster per layer than the dequant fallback - and on Apple Silicon through
the QuixiCore-Metal kernels. A weight *format* is just bits and a scale
layout; there is nothing about it that belongs to one vendor's tensor
cores. FP8 on these 3090s (trick #2) is the same argument one generation
earlier: the format shipped years after the silicon, and the kernels close
the gap.

SlimServe is also not a standalone product: it is the **inference-serving
component of SovereignStack**, our stack for running frontier-class AI
entirely on hardware you own. The premise of SovereignStack is that
serious AI capability shouldn't require renting it from a hyperscaler -
and the premise only holds if the serving layer can extract everything the
owned hardware has. A rack of RTX 3090s serving a 125B-parameter model at
1,200 tok/s with 262K context and half a terabyte of conversation cache is
that premise made concrete: this is what sovereignty costs, and it is a lot
less than people think.

---

## 1. GPU-to-GPU DMA on GeForce cards (+36-58%)

GeForce cards ship with peer-to-peer DMA disabled in the driver, so
tensor-parallel all-reduce bounces every tensor through system RAM. On an
8-way TP model that is fatal: our bring-up numbers were capped by
collectives, not compute.

We run [QuixiAI's patched open-gpu-kernel-modules](https://github.com/QuixiAI/open-gpu-kernel-modules),
which re-enables large-BAR P2P on GeForce (with a per-boot 32 GiB BAR1
setup, `iommu=pt`, and `NCCL_P2P_LEVEL=SYS` to allow P2P across root
complexes on our dual-socket EPYC). Measured on this exact profile:

- c1: 109.6 -> 148.8 tok/s (**+36%**)
- c8: 409.7 -> 647.9 tok/s (**+58%**)

The stock driver still works - the profile only recommends the patched one -
but more than a third of the machine is on the table without it.

## 2. FP8 weights without FP8 hardware

SM86 has no FP8 tensor cores, and Triton's FP8 MoE path wants SM89+. The
experts instead run **Marlin W8A16 block-FP8 kernels**: weights stay in FP8
in memory (the bandwidth win is what matters for a 6B-active MoE), decoded
in-register to bf16 compute.

One subtlety that cost us a debugging session: **expert parallelism is a
correctness requirement here, not a tuning choice.** The MoE intermediate
size is 640 and the FP8 block scales are 128x128 - a tensor-parallel shard
of 640/8 = 80 cannot carry exact block scales. EP keeps 64 whole experts
per rank (640 = 5 x 128) and the math stays exact.

## 3. 51 GiB of embedding tables that never touch the GPU

The model's n-gram "PLE" tables are 47.7 GiB - two full 3090s of VRAM - but
each token reads only 16 rows (2,560 bytes). We pin the table **once in a
/dev/shm segment shared by all eight ranks** and gather those rows over UVA
from inside the forward pass. The gather is CUDA-graph capturable, so it
costs nothing extra under graph replay, and replicating the table per rank
removed a per-step embedding all-reduce this no-NVLink box could not afford.

GPU memory holds weights and KV. Nothing else.

## 4. CUDA graphs, and the capture-size trap

Decode runs under `FULL_DECODE_ONLY` CUDA graph capture. The trap: with MTP
speculation at k=2, every decode step is `num_seqs x 3` query tokens, so the
graph capture size must be at least **3x max_num_seqs** - ours is 96 for 32
sequences. Undersize it and the largest batches silently fall back to eager:
we measured that cliff at **425 tok/s eager vs 880 graphed** at c32 before
pinning the rule down. If your throughput curve has a mysterious dent at
high concurrency, check whether your biggest batches are actually replaying
graphs.

## 5. Speculative decoding with the model's own drafter

Flash-Next ships a single-layer QSA MTP head; we run it at k=2 with index
sharing across MTP iterations. We also built dynamic-k (k=3 below the
concurrency crossover, k=2 above - worth +19.5% at c4 for -2.6% at c32) and
shipped it disabled: c32 is this profile's operating point, and the peak
belongs to static k=2.

One measurement-methodology note: never benchmark speculation with synthetic
repeated-token prompts. Acceptance collapses to ~0% on degenerate text and
looks exactly like a serving bug. Use natural text.

## 6. The 27-slot ceiling, and other scheduler archaeology

At 32 concurrent requests the KV meter read 100% while the scheduler
silently capped at 27 running sequences. The cause: this hybrid's packed KV
slab charges every request ~13-14 blocks at chat context (1,056-token QSA
blocks, a compressor ring block, GDN state blocks at k=2, plus the drafter's
cache group), and the default 0.9 GPU-memory utilization sized a pool that
could not hold 32 of those. Raising utilization to 0.96 (the profile's
value, with the activation and graph reserves re-profiled) fixed it:
**+29% at c32** from scheduling alone.

On top sits admission control: an ASGI middleware caps in-flight generation
requests at 96 (3x the decode slots) and answers 429 + Retry-After beyond
that. Measured at 64 concurrent sessions, oversubscription past 32 buys
zero throughput (780.5 vs 779.7 tok/s) and only queueing delay (TTFT p50
40.2 s vs 11.6 s) - the cap sheds pathological load without policing bursts.

## 7. Exact bf16 KV - because quantized KV was changing answers

We originally shipped TurboQuant-compressed main KV for its 2.25x KV
capacity. Then multi-turn agentic testing surfaced something worse than a
perf bug: the model would occasionally answer the *previous* turn's
question. With only 6B active parameters, this model appears genuinely
sensitive to KV precision.

Main KV is now **always exact bf16** on this platform, enforced by a
registry test, at the model's native 262,144 context. Measured cost:
nothing - bf16 is ~8% *faster* at c1 (no dequant on the attention path) and
at parity c8-c32. The attention kernel had to follow the dtype: the QSA
sparse-gather kernel's tile profile is KV-format-dependent - narrow
16-column tiles keep in-register dequant alive for quantized KV, wide tiles
suit bf16 - and picking the wrong profile collapses long decode steps by
3.8-23x depending on format. Draft-model KV stays TurboQuant-compressed everywhere:
rejection sampling verifies every draft token against the bf16 target, so
draft precision can only affect acceptance rate, never output content.

## 8. Prefix caching on a hybrid model (the default is silently OFF)

vLLM silently disables prefix caching for hybrid (linear-attention) models.
We shipped that default without noticing, and paid full-history re-prefill
on every chat turn: a deep conversation degenerated the whole engine. On
agentic traffic - long shared prefixes extended turn over turn, which is to
say ~99% of our production load - prefix caching approaches a 100% hit rate
and its absence is catastrophic, not marginal.

The hybrid "align" cache mode makes it work: the engine snapshots the
linear-attention state at block-aligned boundaries so a cached prefix can
resume with both its attention KV *and* its recurrent state. It is now
policy, enforced by tests: every profile states prefix caching explicitly.

## 9. The CPU-offloaded KV tier: 4.8 million tokens of warm conversations

The GPU pool at 0.96 utilization holds ~270K tokens - barely one
max-length conversation. Production is many agents with deep histories, so
we built a second tier: **64 GiB of pinned host RAM per rank (512 GiB
total, ~12,150 block slots, ~4.8M tokens)** behind the GPU pool.

The design rides the engine rather than fighting it:

- **Offload is free.** KV blocks are immutable once filled; a dedicated copy
  stream DMAs each newly-filled block to its pinned slot off the critical
  path. A trajectory-centric index maps hash-chained prefixes to slots.
- **Hybrid state is the hard part.** You cannot snapshot linear-attention
  state whenever you like - it is updated in place and covers an unaligned
  token count at any given moment. Every naive scheme we tried (copy the
  live block at finish, copy the penultimate block, freeze-previous during
  prefill) produced subtly or spectacularly garbled resumes. The correct
  source is the engine's own align-mode machinery: when a request crosses a
  block boundary, the boundary state is frozen, hashed into the prefix
  cache, and never written again. The tier looks that immutable snapshot up
  by its boundary hash and saves *that* - torn copies become impossible by
  construction, and restores land exactly where the runner's state-index
  seed expects them.
- **Resume points exist at chunk ends.** Frozen states only materialize
  where a scheduling chunk ended (~2K tokens apart), so the saver scans down
  to the deepest boundary every state group actually has; the remainder
  re-prefills on resume.
- **Byte-verified, then behavior-verified.** A SHA-checking debug mode
  proved every offload/restore round-trip bit-exact; a marker-recall battery
  proved semantics: conversations at 8K/24K/42K depth, fully evicted from
  the GPU pool by churn, resumed **in 0.5 s versus 9.0 s of cold prefill at
  42K** - with restored outputs byte-identical to GPU-cache-hit controls.
- **And it costs nothing.** With the tier live, the deployed service
  measures c1 ~140 / c8 600.5 / c32 up to 1,199.8 - the c32 peak was set
  *with the tier enabled*.

An agent returning to a conversation the GPU evicted an hour ago gets its
first token back in under a second.

## 10. Thinking, tools, and sane defaults

The serving policy is fixed and test-enforced: automatic prefix caching,
automatic tool calling, and thinking always on; never greedy sampling
(benchmarks run the model's shipped sampling defaults, seeded). The chat
template defaults to `reasoning_effort: low` - the shipped default of xhigh
spends hundreds of thinking tokens on trivial turns - and any request can
override it back up when the task deserves deep reasoning.

---

## The bottom line

| | Bring-up | Final |
| --- | ---: | ---: |
| Peak aggregate | 409.7 tok/s | **1,199.8 tok/s** |
| Context | 131,072 | **262,144 (native)** |
| Main KV | quantized | **exact bf16** |
| KV capacity behind the pool | - | **~4.8M tokens in RAM** |
| Multi-turn correctness | tracking errors | **validated clean** |

Every configuration in this post is captured as a SlimServe profile
(`qwen38fn-fp8-8`), with every setting annotated with the measurement that
put it there, and the raw benchmark logs archived alongside. That is the
real lesson of the campaign: none of these tricks is exotic on its own -
the throughput came from measuring one variable at a time, keeping the
receipts, and refusing to call the process started until the workload that
exposed the last failure passed.

*The profile (`qwen38fn-fp8-8`), kernels, KV tier, and benchmark harness
from this post are all in the SlimServe repository - vLLM's engine, ds4's
interface, QuixiCore's kernels, serving as the inference layer of
SovereignStack.*
