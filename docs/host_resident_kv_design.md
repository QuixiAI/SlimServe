# Host-resident main KV for Qwen Sparse Attention (three-tier redesign)

Status: DESIGN + milestone 1 in progress (2026-09-06). Target profile:
`qwen38fn-nvfp4-4` (nvidia/Qwen3.8-Flash-Next-NVFP4 on 4x RTX 3090), then
every Qwen4Exp record.

## Problem

The tiers as built (`HostTierConnector`: pinned-host arena + NVMe file) store
*finished or evicted* conversations and restore them into the GPU pool when a
request resumes. Every *active* request must still fit entirely in GPU KV,
because the attention kernels read device memory, and vLLM enforces it at boot
(`_check_enough_kv_cache_memory`: the pool must hold one max-length request).

On 4x 24 GiB cards the NVFP4 checkpoint leaves no such pool:

| per rank (TP4, EP)                        | GiB   |
| ----------------------------------------- | ----: |
| weights (measured, `Model loading took`)  | 20.59 |
| non-torch (measured)                      |  0.5  |
| peak activation (measured, 16 seqs)       |  0.98 |
| CUDA graphs (capture 48)                  |  0.28 |
| KV pool at utilization 0.97               |  0.77 |
| KV needed for one 262,144-token request   |  2.02 |

Native context is impossible with GPU-resident KV. It is not impossible with
host-resident KV, because of what QSA reads.

## What QSA actually reads

Per query, Qwen Sparse Attention attends to the indexer's top `indexer_budget`
= 2,048 tokens, selected by scoring a compressed cache (1 entry per
`indexer_compress_ratio` = 4 tokens, `indexer_head_dim` 128, 1 head). The
gather kernel (`ops/qsa.py::_qsa_sparse_paged_gqa_splitk_kernel`) touches
2,048 x (K 256 B + V 256 B) fp8 per layer per query, addressed through the
request's block table.

Per-rank bytes per token in the packed slab (11,980,800 B per 1,600-token
block at TP4; 7,488 B/token):

| component                              | B/token | GPU-resident needed? |
| -------------------------------------- | ------: | -------------------- |
| main QSA KV, 12 full-attention layers  |   6,144 | NO - gathered top-k  |
| indexer compressed cache               |    ~770 | yes (dense scan)     |
| raw-key ring + GDN state (amortized)   |    ~570 | yes (recurrent)      |

So a 262,144-token request needs ~0.35 GiB of GPU-resident state and 1.5 GiB
of main KV that can live in pinned host RAM and be read over PCIe by the
gather - the same mechanism the 47.7 GiB PLE table already uses on this stack
(`_ple_host_gather_kernel`: an int64 base address cast to a Triton pointer,
reading cudaHostRegister-ed memory from inside the forward, graph-capturable).

## Design

### 1. Block pointer table (kernel)

The gather kernel stops computing `k_cache_ptr + page * stride_k_block` and
instead loads a per-physical-page base address from `block_base_ptr[page]`
(int64, device tensor, one entry per page of the main-KV slab) and casts it to
a pointer. A page's base is either a GPU row of the hot window or a row of the
pinned host arena. K and V of one page share a base (page layout unchanged:
[block_size, kv_heads, 2*head_dim] per layer, K then V in the last dim).

Cache writes (`reshape_and_cache_flash`) are replaced by a Triton store that
resolves the same pointer table, so a page can be written wherever it lives.
Writes only ever target the request's tail block, which the residency
manager keeps on the GPU (a demoted page is full and immutable).

Cost model (PCIe gen3 x16, ~12 GB/s measured host->device): 12 MiB per query
token per step when every selected token is host-resident:

| concurrency (MTP k=2, 3 queries/req) | extra/step | vs today's step |
| ------------------------------------ | ---------: | --------------- |
| c1                                   |   ~3.5 ms  | +15%            |
| c8                                   |    ~25 ms  | ~2x             |
| c16                                  |    ~50 ms  | ~3x             |

Measured (milestone 1, 2026-09-06, RTX 3090 = PCIe 4.0 x16; real kernel,
TP4 geometry, fp8, top-k 2048 uniformly random over 262,144 tokens, x12
layers per step; `perf/results/2026-09-06/qwen38fn-nvfp4-4/bench_ptrtable_gather_v4.log`):

| rows (queries) | slab kernel | table, all GPU | all host | 50% host | oldest 75% host |
| -------------- | ----------: | -------------: | -------: | -------: | --------------: |
| 3  (c1)        |    1.45 ms  |   1.51 ms      |  1.83 ms |  1.49 ms |  1.50 ms        |
| 24 (c8)        |    1.73 ms  |   2.13 ms      | 12.05 ms |  6.18 ms |  9.13 ms        |
| 48 (c16)       |    3.21 ms  |   4.20 ms      | 23.9 ms  | 12.2 ms  | 18.0 ms         |
| 96 (c32)       |    5.74 ms  |   7.09 ms      | 47.5 ms  | 24.1 ms  | 35.7 ms         |

The host gather runs at ~25 GB/s effective (PCIe 4.0 saturated); parity is
bit-exact in every configuration. The pointer-table path costs ~20% on
GPU-resident pages (int64 table + sentinel test) - removable later by
making it the only path. Single-stream cost is negligible; at c8 an
all-host request adds ~10 ms per step, so the hot window and the page
selection statistics of real traffic (not uniform, as here) set the batch
regime's cost.

### 2. Residency manager (worker)

The main QSA KV leaves the packed slab and becomes its own slab with two
physical backings: `gpu_rows` (the hot window, sized from what is left after
weights) and `host_rows` (the existing pinned arena, registered PORTABLE |
MAPPED). The scheduler's block pool is sized to `gpu_rows + host_rows_active`
where `host_rows_active` is a reserved share of the arena for active
requests (max_num_seqs x max_model_len / block_size blocks), the rest of the
arena staying the trajectory store it is today.

A logical block id maps to one physical row through the pointer table. Rules:

- A block is born on a GPU row (it is about to be written).
- When no GPU row is free for a new block, the manager demotes the coldest
  *full* block of any running request: copy GPU row -> host row on the copy
  stream, then rewrite that block's pointer-table entry on the compute stream
  after the copy's event, so no kernel observes a half-moved page and no host
  sync is needed.
- Coldness: oldest-full-first within a request, requests round-robin
  (measurable later: the indexer's selections could rank pages by hit count).
- Prefix-cache hits on host-resident blocks need nothing: they are readable.
- The block being written this step (from `slot_mapping`) must be GPU-resident;
  a freed logical block reissued by the scheduler is rebound to a free GPU row
  before its first write (no copy, content is new).
- Demotion never targets a block referenced by this step's `slot_mapping` or
  the ring/indexer/GDN groups (which stay in the packed slab, always GPU).

### 3. Tier integration (scheduler)

The trajectory index already records every full attention block's host slot at
fill time (write-through). With the pointer table, a tier *restore* becomes a
rebind: the resumed request's logical blocks point at the arena slots that
already hold the bytes - zero-copy resume, no GPU pool pressure. Promotion
from NVMe fills arena slots exactly as today and then rebinds. Demoted rows of
active requests are the same physical objects as trajectory slots, so:

- an arena slot bound to an active logical block is pinned against tier LRU;
- when the request finishes, the slot simply stays as the trajectory's copy.

`_check_enough_kv_cache_memory` counts `gpu_rows + host_rows_active` pages for
the main-KV group; the packed slab's own check (indexer/ring/GDN) is unchanged
and is what actually bounds context on this hardware (~0.35 GiB per max-length
request).

### 4. What stays exactly as it is

fp8 main-KV format and the in-register e4m3 decode; the indexer and its
compressed cache; GDN align-mode state contract and tail snapshots; the packed
slab for every non-main-KV group; the NVMe write-through/demotion/promotion of
trajectories; CUDA-graph capture (pointer loads are ordinary global loads; the
table is a static device tensor).

## Milestone 3 status (2026-09-06)

Boots at max_model_len 262,144 on 4x RTX 3090 (`qwen38fn-nvfp4-4`):
attention block 12,688 tokens (sized from the indexer page), packed stride
10,556,416 B, a max-length request needs 0.37 GiB of GPU KV, pool 441,505
tokens (1.68x one max-length request), main KV 832 rows x 6.5 MB with 24
GPU hot rows and 5.03 GiB of pinned host rows per rank. Two structural
findings shaped the implementation: the CSA+linear grouping padded every
mamba page to the main-KV page (8x), and the aligner override has to be
sticky on cache_config because draft config views re-run it.

## Milestone 4 status (2026-09-06): tier rebind

The trajectory tier gained a second pinned arena of main-KV slots (one
scheduler block of sub-rows per slot, `main_kv_tier_gb_per_rank`). A slot
is reserved when the scheduler allocates the block (`main_homes` in the
connector metadata), so every demotion the residency performs lands in the
tier directly; when the block fills, the connector emits `main_flush` and
the residency copies its still-resident dirty rows into the slot (rows stay
hot), confirmed one build later like slab writes. A tier hit REBINDS the
resumed request's new block ids onto the trajectory's slots
(`main_rebinds`: an offset-table update, no copy) and pins them until the
request finishes; slots reserved for blocks that never filled are released
(`main_release`) so their rows fall back to the residency pool. Live: hits
after full-pool eviction, recall correct at 8K/24K/42K, 42K resumes 2.5x
faster than cold. Not yet: main slots on NVMe (trajectories carrying them
are deleted, not demoted, under host pressure).

## 8-GPU FP8 port (2026-09-06): the block floor and what TP changes

- `qwen38fn-fp8-8/rtx3090` runs the same design (extras: host_tier 12,
  main_kv_tier 48, gpu_rows 104, sub_blocks 8 -> 13). The main KV is one KV
  head replicated per rank, so a block is 84,451,328 B per rank at TP8 as at
  TP4; only the GDN state page halves (406 KB).
- The aligner's natural block (GDN page / 64 B per token) therefore halves
  to 6,352 = 16 x 397 tokens, which splits into no sub-row near the
  requested 8 (only 1 or 397), making every hot row a whole 19.5 MB block
  and failing the one-request memory check. `main_kv_block_tokens`
  (default 12,688) holds the block at the validated 13 x 976 geometry and
  pads the GDN page to match (interface.py; tests/v1/core/
  test_main_kv_planner_helpers.py). A max-length request still charges 38
  packed blocks (21 indexer + 1 ring + 16 GDN) at the 10.56 MB stride.
- Result: 3.65 GiB available KV = 306 packed blocks, 2,110,949-token pool,
  8.05 max-length requests (1.89x on the 2026-09-02 record). Exact bench
  c1/c8/c16/c32 116.2/549.4/631.3/787.5 vs 131-136/547-585/838/880: c1 and
  c16 pay the residency's per-step host bookkeeping (10.6 ms/step at c1),
  c32 runs 21 requests because a chat request charges 14 slab blocks (12
  of them GDN snapshots in attention-sized slots).
- Owed, in order: (a) incremental `prepare_step` (device tables updated
  only for changed rows, pinned staging); (b) per-group slab strides so
  GDN snapshots stop paying an attention-sized slot (21 -> 32 chat
  requests, ~8x -> ~13x max-length residency in the same pool); (c) the
  reclaim hazard - CLOSED 2026-09-07: `KVTierIndex._delete` freed main
  slots that pool blocks still pointed into (rebinds, demoted homes), so
  a block alive in the GPU prefix cache could read another trajectory's
  bytes after reclaim. The index now carries block-level holds
  (`hold_main` / `unhold_main`) staged with every home and rebind;
  `_main_busy` honours them, and they drop only when the pool reuses the
  block id, a confirmed flush moved the rows into a new slot, or the
  reservation is released. A fresh block that finds the tier full is
  un-homed so it never demotes into a stale slot; (d) milestone 4b (main
  slots on NVMe).

## Multi-pool packed slab (2026-09-07, E5b)

- Problem: the packed slab hands every group the same block stride, so a
  GDN state page (5.0 MB bf16 for a 12-layer group, 0.2 MB for the 1-layer
  group) and the compressor ring (29 KB) each occupy a 10.56 MB slot. A
  chat request holds 14 slots: 1 attention, 1 ring, and per GDN group the
  running state plus `num_speculative_blocks` (= k) rollback slots.
- Design: one BlockPool per page-size class (`KVCacheConfig.pool_num_blocks`
  / `group_pool`, `KVCacheTensor.pool`, `KVCacheBlock.pool_id`). Pages of
  the non-attention classes are laid out at their natural size (the
  aligner's padding is dropped for them). The coordinator builds one pool
  per class and gives each group's manager its pool; capacity checks,
  frees, cache-hit lookups, events and the tier's boundary-state pins are
  per pool. The worker allocates one backing per pool; the tier connector
  registers every backing and the DMA selects the view by group id; the
  arena row stays the attention stride.
- Sizing (`kv_connector_extra_config.kv_pool_deep_requests`, 0 = single
  pool): the attention-class pool holds that many max-length requests
  (+ null + 1); the remaining memory goes to the other classes at equal
  chat-context concurrency (per-class demand = 1 for attention/ring,
  1 + k for each GDN group).
- Result on qwen38fn-nvfp4-4 (bf16 GDN state): pools [44, 71, 25, 9] at
  [10.56 MB, 5.03 MB, 225 KB, 29 KB] = 0.77 GiB, deep capacity 2.1x
  (unchanged), chat concurrency 8 (from 5): c8 +17%, c32 +27%, c1
  unchanged; restores/recall/canary pass.
- Limit: at 8 running the rollback slots are 8 x 4 groups x 2 x 5 MB =
  320 MB; going past ~8 running needs fewer per-position state copies
  (recompute-on-reject) or a smaller state, not a layout change.

## Milestones and gates

1. **Pointer-table gather kernel + microbench** (this entry). Gate: bit-exact
   parity against the base-pointer kernel with pages on the GPU, on the host,
   and mixed; measured step cost at rows 3/24/48/96 x top-k 2048 for 0/50/100%
   host-resident pages. Decides the hot-window policy.
2. **Pointer-table cache store** (Triton) + parity with `reshape_and_cache_flash`.
3. **Residency manager + split main-KV slab** in the worker; boot with the
   scheduler pool = GPU rows + active host rows; `qwen38fn-nvfp4-4` at
   max_model_len 262144 reaches health.
4. **Tier rebind**: zero-copy restore/promotion; eviction-restore acceptance
   with a tiny GPU window so most of every active request is host-resident.
5. **Serving gates**: exact-token bench c1/c8/c16, deep-context marker recall
   (8K/24K/42K/100K/200K), multi-turn tracking, image request; then the
   8-GPU FP8 record gets the same design (its 1.03x pool becomes many x).

## Zero ops carry their KV group (2026-09-08 fix)

The tier restore zeroes the resumed request's compressor ring block
defensively. Under the multi-pool packed slab (E5b) every page-size class
is its own pool with its own block ids, so a zero op is `(block, group)` and
the DMA zeroes `_blocks_for(group)[block]`. The previous bare block id was
zeroed in pool 0 - the attention slab - which under the single slab was the
same physical row (harmless) and under the multi-pool slab was attention
block N: a mixed resume (GPU prefix-cache hit on the leading block, tier
restore of the rest) whose ring block id coincided with the cached block's
id lost that block's indexer keys, and the top-k never selected the prompt
head again. Pure tier resumes rewrite every restored attention block after
the zero, which is why only mixed resumes failed. Regression test:
`test_zero_op_targets_the_groups_pool_not_pool_zero`. Standing gate: the
chat oracle with identical back-to-back prompts (it recycles the same few
block ids, which is what makes the collision reliable).

## Row validity in the residency (2026-09-08 fix)

A residency row has up to three copies: the GPU hot-window row, a pool
host row, and a tier-slot row. `dirty` says the GPU copy is newer than any
host copy; `flushed_to` says WHICH host address holds the current copy
(-1: none). Every copy decision uses both: flush and demotion copy when the
row is dirty or its current copy is not at the destination; bind resets
`flushed_to` (the GPU copy is about to diverge); rebind sets it to the slot
row; `set_home` moves already-demoted rows unconditionally (they are never
dirty). The previous rule ("clean means already in the home") was only true
for the slot a row was first flushed to, and broke the moment the tier
connector re-homed a GPU prefix-cache hit into a resumed lineage's slot: the
new slot received only the demoted rows, and the resumed conversation read
stale bytes for its prompt head (the 2026-09-06 "empty follow-up" episode
and the 2026-09-07 oracle failures). Regression test:
`test_rehomed_block_copies_its_clean_rows_into_the_new_slot`.

`VLLM_KV_TIER_VERIFY=1` now also digests every main-KV slot at flush and
at rebind ("main-kv verify: ... OK|MISMATCH") beside the slab digests.

## Scaling campaign (2026-09-07): throughput vs concurrency on 4x RTX 3090

Measured on qwen38fn-nvfp4-4 (perf/results/2026-09-07/scale/). The step at
8 running is 29 ms; at 16 running, with every page hot and in-graph, 44 ms
busy (census): the O(rows) terms are (a) the int8 skinny route's row cutoff
(above 32 rows the dense projections fall back to per-call dequant plus
cuBLAS: +8 ms), (b) the RING_LL collectives on doubled messages (+4 ms;
LL128 recovers it but is barred on PCIe by NCCL policy), (c) Marlin MoE
saturating (+2 ms, expected). The sparse gather is flat once the hot
window holds the active rows (48 rows for 16 chat requests).

Per-request VRAM at chat context is what caps the running count: ~26 MB
of hot main-KV rows (4 x 6.5 MB), 45 MB of GDN state (3 groups x (1 + k)
slots x 5.03 MB), 10.6 MB indexer keys, 0.7 MB ring/PLE - ~83 MB, so 32
hot requests need ~2.6 GiB against a 1.15-1.5 GiB budget.

### GDN single-slot replay (design, not built)

The k+1 SSM slots per group exist only because acceptance is unknown until
sampling: the fused recurrent kernel reads the previous step's state from
slot[accepted-1] and writes the state after every draft position into its
own slot; the conv state already lives in ONE slot with k extra history
columns rolled by the accepted count (causal_conv1d_update,
num_accepted_tokens). Replace the SSM rollback slots with a replay:

- Keep one SSM slot per group per request holding the state BEFORE the
  previous step's tokens (state_prev).
- Keep the previous step's per-token kernel inputs for the k+1 positions
  (post-conv q/k/v, g, beta: ~17 KB per token per layer; 36 layers x 32
  rows x 3 tokens = 57 MB per rank, a per-batch-row scratch).
- At the next step, with a_prev known, run the recurrence over
  T = a_prev + (k+1) tokens from state_prev: the first a_prev tokens are
  the replayed accepted prefix (outputs discarded), the rest are this
  step's draft. The kernel stores the state ONLY after position a_prev-1,
  in place, as the new state_prev (a REPLAY constexpr: store at one index
  instead of every position). One launch per layer, T <= 6 instead of 3.
- Planner: num_speculative_blocks 0 for the SSM/PLE groups (chat demand 1
  instead of 1 + k), so a chat request charges 3 GDN blocks (15 MB) not 9
  (45 MB): 32 running requests need 0.48 GiB of GDN state instead of
  1.45. The align-mode postprocess copy (accepted slot -> running slot)
  disappears with the slots.
- Cost: the replay adds up to k tokens of recurrence per layer per step
  (~0.6 ms at 8 running, ~1.5 ms at 32) and the scratch write of the
  kernel inputs. Gates: bit-exact parity against the multi-slot path on
  recorded (inputs, acceptance) traces; deep recall; the tier's tail
  snapshots are unchanged (they save the running slot at block
  boundaries).
