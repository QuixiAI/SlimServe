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
