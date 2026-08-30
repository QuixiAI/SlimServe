# DSV4 bf16 main KV on Ampere (A100)

Operator directive 2026-08-29: bf16 main KV is the serving policy; the DSV4
A100 profiles must serve it. This is the design pass for the kernel/stack
work. The fp8_ds_mla path stays intact and qualified; bf16 is added beside
it and flips the profiles only after the gates below.

## Why this is an instantiation-and-wiring job, not a new kernel

Nearly every needed piece already exists in-tree; the fp8-only state was an
unwired path, not a missing capability:

- The sparse-MLA decode kernel is one template,
  `mla_decode_fp8_v<SPARSE, PART, QW, VW, NFP8, SMODE>`
  (`csrc/quixicore/serving/mla_kernels.cuh:357`). `NFP8` is the count of
  leading fp8 elements per slot; `SLOT_BYTES = NFP8 + 2*(QW-NFP8)`.
  **`NFP8=0` means an all-bf16 slot and is the geometry GLM-5.2-Vision
  ships on A100 today** (`py_mla_decode_bf16_sparse_glm`,
  `tm_cuda_serving.cu:1366`: "this is the geometry that actually runs on
  A100"). The generic scalar read path handles `NFP8=0` with the scale
  plane never dereferenced.
- DSV4's production decode is the merged two-source persistent variant of
  the same template (`launch_mla_decode_fp8_sparse_dsv4_merged`,
  `tm_cuda_serving.cu:1546`), instantiated only at the default DSV4
  geometry `<true, true, 512, 512, 448, 0>` (448 fp8 nope + 64 bf16 rope,
  576B data + 8B UE8M0 scales). Page stride and scale offsets are already
  RUNTIME arguments.
- The KV write path already has the bf16 branch:
  `fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert`, selected
  by `cache_dtype == torch.bfloat16` in `_fused_qnorm_rope_kv_insert`
  (`models/deepseek_v4/attention.py:855`). The TQ draft path already uses
  this very kernel as its bf16 transform.
- The compressor supports plain rows: `store_full_kv = head_dim == 512 and
  kv_cache.dtype != torch.uint8` (`models/deepseek_v4/compressor.py:447`).
- The KV spec path supports plain rows: `get_kv_cache_spec` emits a
  natural-element-size `MLAAttentionSpec` when the dtype is not
  fp8_ds_mla (`attention.py:~895`); the packed-slab stride recomputes from
  group specs, so the HostTierConnector needs no change.
- `_resolve_dsv4_kv_cache_dtype` already has the plain-row resolve branch
  (auto/bfloat16 -> bf16 rows); it is simply not reachable on Ampere
  because the Ampere class pins `use_fp8_ds_mla_layout = True`.

The genuinely new work is three seams: a bf16 instantiation of the merged
decode, a bf16-source variant of the prefill gather, and the Python gating
that lets the Ampere class run the plain-row layout.

## Page format

bf16 slot = 512 contiguous bf16 elements (448 nope + 64 rope) = **1024 B**,
no scale plane. Both DSV4 caches flip together: the SWA cache and the
compressed (C128) cache. Unchanged: DSpark TurboQuant draft KV (policy
exemption), and the Lightning-indexer K cache (its own fp8 encode; it
selects tokens, it does not produce attention output — same content-safety
argument as draft KV, and its logits kernel `fp8_paged_mqa_logits` is
selection-only).

## Changes by layer

1. **Decode (csrc)** — add the `<true, true, 512, 512, 0, 0>` instantiation
   to the merged launcher; binding gains a `bf16_pages: bool` arg (or reads
   `main_cache.dtype == bf16`). Scale-offset args pass 0; the NFP8=0 path
   never reads them. Slot addressing: `SLOT_BYTES` is compile-time per
   instantiation, page stride stays runtime. The partial reducer is
   VW=512 and unchanged.
   Stage 2 (measured, optional): a `VECBF16` fast path mirroring `VECDSV4`
   — uint2 8B/lane rounds instead of the scalar 2B/lane generic path — if
   the A/B shows the generic path leaving decode bandwidth on the table.
2. **Prefill gather** — `dequantize_and_gather_k_cache`
   (`models/deepseek_v4/common/ops/cache_utils.py:414`) gains a bf16-source
   branch: pages are already the workspace dtype, so it is a paged gather
   copy (no dequant). Same signature, dispatch on `k_cache.dtype`.
3. **Python gating** — the Ampere attention class computes
   `use_fp8_ds_mla_layout` from the requested dtype instead of pinning it:
   fp8* -> True (existing path, byte-identical), auto/bfloat16 -> False
   (plain rows). `_resolve_dsv4_kv_cache_dtype` then resolves bf16 via its
   existing branch and the insert/compressor/spec seams follow their
   existing dtype branches. `forward_mqa`/`_forward_decode` pass the
   dtype-matched cache pointers and strides to the decode binding.
4. **Profiles** — after gating, flip the five DSV4 A100 records to
   `kv_cache_dtype: auto` and re-enforce a100 in
   `test_no_profile_quantizes_main_kv` (drop the dsv4-flash exemption).
   KV pool budgets: bf16 rows are 1024B vs 584B — the q4ktail-2 explicit
   byte budgets hold ~57% of the fp8 token count; requalify pool sizes in
   the same pass.

## Cost accounting (to be measured, not assumed)

Decode-side KV reads grow 584 -> 1024 B/token (~1.75x attention bytes).
MLA was measured at ~48% of A100 decode before the partitioned kernel
work, so the naive ceiling is a noticeable single-digit-to-low-teens %
decode regression at long context, partially hidden by the persistent
partitioning. The A/B against the fp8 baseline (currently being collected
by the WildChat sweep) decides whether bf16 becomes the default or stays a
quality option. Pool capacity halves at equal bytes; deep-context
concurrency shifts to the host KV tier, same as the rtx3090 rollout.

## Validation gates

1. Kernel parity: flip-aware mean-rel harness (per the established
   methodology) — bf16 merged decode vs fp32 reference, and vs the fp8
   path on identical synthetic pages; both sources (SWA + compressed),
   partitioned and not, slot-indexed and block-table-indexed.
2. Boot gate: `dsv4-q4ktail-4 --serve` with auto KV reaches health; graph
   capture (FULL_DECODE_ONLY capture-64) intact.
3. Correctness: marker recall at depth (deep-context WildChat harness) and
   needle checks; byte-plausible outputs vs fp8 control at temperature-0
   diagnostic sampling is NOT expected (different KV rounding) — the gate
   is recall integrity and the slimserve-canary quality probe, not
   bit-equality.
4. Perf: exact-token TP4/TP8 c1/c8 A/B vs the 2026-08-29 fp8+tier baseline;
   TP-scaling sanity holds (TP4 >= 1.5x TP2).
5. Tier: host-tier acceptance re-run on bf16 pages (block stride changes).
