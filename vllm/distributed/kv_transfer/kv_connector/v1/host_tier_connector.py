# SPDX-License-Identifier: Apache-2.0
"""Host KV tier: pinned-RAM second tier for the packed KV slab.

A KV connector that mirrors trajectories into a pinned host arena and
restores them when a request's prefix misses the GPU cache but matches a
stored trajectory. Built for SlimServe's packed cross-layer slab, where one
block-id is one contiguous stride region, so every transfer is a single DMA
per block per rank.

Contract (matches the engine's hybrid semantics, learned the hard way):
- Full-attention blocks are immutable once full: offloaded at fill time
  (one step behind the verified-token watermark so speculative rejections
  can never be captured), restored across the whole resumed span.
- Mamba align-mode state exists only at the RESIDENT tail boundary
  (earlier positions are nulls), so state blocks are offloaded once, when
  the request finishes, via the async-save path
  (request_finished_all_groups -> True; blocks stay alive until the worker
  reports the save complete). A trajectory is therefore resumable exactly
  at its recorded tail boundary - the natural shape of append-only agentic
  traffic.
- Ring (CircularBufferSpec) groups are rolling per-request state, excluded
  from prefix caching by their own manager and reconstructed by
  allocate_external_computed_blocks; the tier ignores them.
- Restores ride the framework's async-KV-load path
  (WAITING_FOR_REMOTE_KVS): issued while the request waits, hidden under
  running decode, and re-queried per scheduling step if clipped.

Config (kv_transfer_config.kv_connector_extra_config):
    host_tier_gb_per_rank: pinned arena size per rank in GiB (required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import _get_packed_kv_cache_layout
from vllm.v1.core.kv_tier_index import HostKVTierIndex
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class HostTierMeta(KVConnectorMetadata):
    # req_id -> [(host_slot, gpu_block_id), ...] restores for held requests.
    restores: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # batch_seq -> [(gpu_block_id, host_slot), ...] attention fill offloads.
    offloads: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    # req_id -> [gpu_block_id, ...] blocks to zero alongside the restore
    # (the ring block, whose framework zeroing was skipped for the
    # async-load range but which the tier deliberately does not restore).
    zeros: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class _ReqTrack:
    # Per-config-group block ids, position-indexed; -1 marks null blocks.
    group_blocks: list[list[int]]
    staged_upto: int = 0
    planned_blocks: int = 0
    planned_attn_slots: list[int] = field(default_factory=list)
    planned_state_slots: dict[int, int] = field(default_factory=dict)
    planned_start: int = 0
    restored_upto: int = 0  # blocks already staged for restore
    # Trajectory key this request stages into: the adopted trajectory's
    # owner when the request resumed from the tier (its true lineage), the
    # request's own id otherwise. Never derived from shared content.
    owner: str | None = None


class HostTierConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config
        tier_gb = float(extra.get("host_tier_gb_per_rank", 0))
        if tier_gb <= 0:
            raise ValueError(
                "HostTierConnector requires kv_connector_extra_config."
                "host_tier_gb_per_rank > 0"
            )
        # Prefer the planner's authoritative packed stride over recomputing
        # from group specs: generate_scheduler_kv_cache_config flattens a
        # UniformTypeKVCacheSpecs group to ONE arbitrary member layer's spec,
        # so on the scheduler side a heterogeneous uniform group (e.g.
        # GLM-DSA: MLA main KV + indexer k_cache pages in one group) yields a
        # spec-derived stride far below the real slab stride. The two roles
        # then disagree on num_slots and the scheduler plans host slots the
        # worker arena does not have (observed: 59811 vs 15339 slots,
        # IndexError at restore). KVCacheTensor.block_stride survives the
        # scheduler deepcopy unmodified, so both roles agree through it.
        packed_strides = {
            t.block_stride
            for t in kv_cache_config.kv_cache_tensors
            if t.block_stride
        }
        if packed_strides:
            assert len(packed_strides) == 1, (
                f"host-tier: mixed packed block strides {packed_strides}"
            )
            self.block_stride = packed_strides.pop()
        else:
            self.block_stride, _ = _get_packed_kv_cache_layout(
                kv_cache_config.kv_cache_groups
            )
        self.num_slots = int(tier_gb * (1 << 30)) // self.block_stride
        if self.num_slots <= 0:
            raise ValueError("host tier smaller than one block stride")

        groups = kv_cache_config.kv_cache_groups
        self.num_groups = len(groups)
        self.attn_groups = [
            gid
            for gid, g in enumerate(groups)
            if not isinstance(g.kv_cache_spec, (CircularBufferSpec, MambaSpec))
        ]
        # Per-request state groups saved once at finish and restored on
        # resume: the mamba align-state groups only. In align mode the
        # engine freezes each boundary state in the pool's prefix cache
        # (cache_blocks at the boundary crossing; migrations only copy OUT
        # of it), so the tail save reads that immutable snapshot - never
        # the live, in-place-updated current block, whose state covers an
        # unaligned token count and desynchronizes a resumer.
        self.state_groups = [
            gid
            for gid, g in enumerate(groups)
            if isinstance(g.kv_cache_spec, MambaSpec)
        ]
        # The QSA compressor ring is NOT saved or restored. The framework
        # never zeroes ring blocks (CircularBufferSpec is not in the
        # zero-recording spec set), so engine-internal prefix hits - which
        # are validated correct - resume on a freshly CLAIMED ring holding
        # stale bytes; this works because a hit lands on an aligned
        # boundary, where the ring's open compression group is empty and
        # nothing unwritten is read. Tier resumes land on the same aligned
        # boundaries, so they are in the same equivalence class. The tier
        # still zeroes the ring block defensively: zero is deterministic
        # and cannot smuggle Inf/NaN into predicated lanes the way another
        # tenant's stale float bytes could.
        self.ring_groups = [
            gid
            for gid, g in enumerate(groups)
            if isinstance(g.kv_cache_spec, CircularBufferSpec)
        ]
        if not self.attn_groups:
            raise ValueError("host tier requires a full-attention group")
        # Token granularity of request.block_hashes == the attention block
        # size AFTER hybrid alignment. cache_config.block_size can hold a
        # stale pre-alignment value in the scheduler process; the group spec
        # is authoritative.
        self.hash_block_size = groups[self.attn_groups[0]].kv_cache_spec.block_size

        if role == KVConnectorRole.SCHEDULER:
            logger.info(
                "host-tier: %d slots, block %d tokens, attn groups %s, "
                "state groups %s",
                self.num_slots,
                self.hash_block_size,
                self.attn_groups,
                self.state_groups,
            )
            self.index = HostKVTierIndex(self.num_slots)
            self._tracks: dict[str, _ReqTrack] = {}
            self._requests: dict[str, Request] = {}
            self._staged_restores: dict[str, list[tuple[int, int]]] = {}
            self._staged_offloads: dict[int, list[tuple[int, int]]] = {}
            self._staged_zeros: dict[str, list[int]] = {}
            # Writes issued last step; confirmed next build (in-order copy
            # stream: any restore issued later observes completed writes).
            self._last_step_write_slots: list[int] = []
            self._offload_seq = 0
            self._block_pool = None
            # State blocks ref-pinned for in-flight tail saves, released one
            # meta build after their copies were issued (same in-order-stream
            # window as slot confirmation). Two-phase: staged -> issued.
            self._pins_staged: list[list] = []
            self._pins_issued: list[list] = []
        else:
            self._dma = None
            self._pending_restore_reqs: dict[int, str] = {}
            self._seq = 1 << 20

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        self._block_pool = gpu_block_pool

    # ==================================================================
    # Scheduler side
    # ==================================================================

    @staticmethod
    def _owner(request: "Request", track: "_ReqTrack | None") -> str:
        """Trajectory key for this request's saves.

        The adopted lineage when the request resumed from the tier, else
        the request's own id. NEVER derived from shared content: keying on
        block_hashes[0] collapsed every conversation sharing a system
        prompt into one chimera trajectory (root cause of the 6-hits-per-
        197-saves production miss rate).
        """
        if track is not None:
            if track.owner is None:
                track.owner = request.request_id
            return track.owner
        return request.request_id

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        bs = self.hash_block_size
        track = self._tracks.get(request.request_id)
        if track is not None and track.planned_blocks:
            # Re-query mid progressive restore: report the remainder.
            remaining = track.planned_blocks * bs - num_computed_tokens
            if remaining > 0:
                return remaining, True
        hit = self.index.lookup(request.block_hashes)
        if hit is None:
            if logger.isEnabledFor(logging.DEBUG) and len(request.block_hashes) >= 8:
                logger.debug(
                    "host-tier: lookup miss for %s: %s",
                    request.request_id[-8:],
                    self.index.explain_miss(request.block_hashes),
                )
            return 0, False
        hit_owner, n_blocks, attn_slots, state_slots = hit
        n_tokens = n_blocks * bs
        # Resume is only possible exactly at the stored tail boundary
        # (mamba state exists only there), and at least one token must
        # remain to compute.
        if n_tokens <= num_computed_tokens or n_tokens > request.num_tokens - 1:
            return 0, False
        if track is None:
            track = _ReqTrack(group_blocks=[[] for _ in range(self.num_groups)])
            self._tracks[request.request_id] = track
        # Adopt the resumed trajectory: this request is its continuation,
        # so its saves extend that lineage instead of duplicating it.
        track.owner = hit_owner
        track.planned_blocks = n_blocks
        track.planned_attn_slots = attn_slots
        track.planned_state_slots = state_slots
        track.planned_start = num_computed_tokens // bs
        track.restored_upto = num_computed_tokens // bs
        logger.info(
            "host-tier: hit for %s: resume at block %d (%d tokens)",
            request.request_id[-8:],
            n_blocks,
            n_tokens,
        )
        return n_tokens - num_computed_tokens, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        track = self._tracks.get(request.request_id)
        if track is None:
            track = _ReqTrack(group_blocks=[[] for _ in range(self.num_groups)])
            self._tracks[request.request_id] = track
        # Full current allocation; keep nulls (-1) to preserve positions.
        track.group_blocks = [
            [(-1 if b.is_null else b.block_id) for b in group]
            for group in blocks.blocks
        ]
        if num_external_tokens <= 0 or not track.planned_blocks:
            return
        bs = self.hash_block_size
        # The scheduler may clip the external span to its budget and
        # re-query per step; stage restores for the newly covered blocks.
        # Progress is planned-relative: request.num_computed_tokens is not
        # advanced while the request waits on the async load.
        covered = min(
            track.restored_upto + num_external_tokens // bs,
            track.planned_blocks,
        )
        ops: list[tuple[int, int]] = []
        for logical in range(track.restored_upto, covered):
            for gid in self.attn_groups:
                gb = track.group_blocks[gid]
                if logical >= len(gb) or gb[logical] < 0:
                    logger.warning(
                        "host-tier: no target block for %s attn g%d pos %d",
                        request.request_id[-8:],
                        gid,
                        logical,
                    )
                    track.planned_blocks = 0
                    return
                ops.append((track.planned_attn_slots[logical], gb[logical]))
        # Tail-boundary state blocks ride the FINAL restore chunk. The
        # resumed state must land at position ``boundary_blocks - 1`` of
        # each mamba group: the worker seeds its running state index as
        # ``(num_computed - 1) // block_size`` and reads exactly that
        # column (mamba_hybrid.MambaHybridModelState.add_request). With the
        # align-mode external allocation shape ([null] * (k-1) + [real])
        # that position is the group's single real block.
        zero_blocks: list[int] = []
        if covered == track.planned_blocks and track.planned_state_slots:
            state_pos = track.planned_blocks - 1
            for tier_gid, slot in track.planned_state_slots.items():
                gid = self.state_groups[tier_gid]
                gb = track.group_blocks[gid]
                if state_pos >= len(gb) or gb[state_pos] < 0:
                    logger.warning(
                        "host-tier: no state target for %s group %d pos %d "
                        "(group blocks: %d, last real: %s)",
                        request.request_id[-8:],
                        gid,
                        state_pos,
                        len(gb),
                        max((b for b in gb if b >= 0), default=None),
                    )
                    track.planned_blocks = 0
                    return
                ops.append((slot, gb[state_pos]))
            # Zero the ring defensively (see __init__: internal hits run
            # on a stale-claimed, never-zeroed ring; zero is the same or
            # strictly cleaner).
            for gid in self.ring_groups:
                zero_blocks.extend(b for b in track.group_blocks[gid] if b > 0)
        if ops:
            self._staged_restores.setdefault(request.request_id, []).extend(ops)
            if zero_blocks:
                self._staged_zeros.setdefault(
                    request.request_id, []
                ).extend(zero_blocks)
            track.restored_upto = covered
            logger.info(
                "host-tier: staged %d restore ops for %s (blocks %d..%d of %d)",
                len(ops),
                request.request_id[-8:],
                track.planned_start,
                covered,
                track.planned_blocks,
            )
        if covered == track.planned_blocks:
            track.planned_blocks = 0  # restore fully staged

    # Offload staging ---------------------------------------------------

    def _absorb_block_allocations(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            track = self._tracks.get(new_req.req_id)
            if track is None:
                track = _ReqTrack(
                    group_blocks=[[] for _ in range(self.num_groups)]
                )
                self._tracks[new_req.req_id] = track
            track.group_blocks = [list(g) for g in new_req.block_ids]
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            new_ids = cached.new_block_ids[i]
            if new_ids is None:
                continue
            track = self._tracks.get(req_id)
            if track is None:
                continue
            for gid, group_new in enumerate(new_ids):
                if gid < len(track.group_blocks) and group_new:
                    track.group_blocks[gid].extend(group_new)

    def _stage_filled_attention_blocks(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        bs = self.hash_block_size
        for req_id in scheduler_output.num_scheduled_tokens:
            track = self._tracks.get(req_id)
            request = self._requests.get(req_id)
            if track is None or request is None:
                continue
            n_full = min(
                request.num_computed_tokens // bs,
                len(request.block_hashes),
                min(
                    (len(track.group_blocks[g]) for g in self.attn_groups),
                    default=0,
                ),
            )
            if track.staged_upto > n_full:
                track.staged_upto = n_full  # preemption reset
                continue
            owner = self._owner(request, track)
            for logical in range(track.staged_upto, n_full):
                block_hash = request.block_hashes[logical]
                ops: list[tuple[int, int]] = []
                for gid in self.attn_groups:
                    block_id = track.group_blocks[gid][logical]
                    if block_id < 0:
                        continue
                    slot = self.index.stage_attention(owner, logical, block_hash)
                    if slot is not None:
                        ops.append((block_id, slot))
                if ops:
                    self._offload_seq += 1
                    self._staged_offloads[self._offload_seq] = ops
            track.staged_upto = n_full

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        confirmed, self._last_step_write_slots = self._last_step_write_slots, []
        self.index.confirm_writes(confirmed)
        if self._pins_issued and self._block_pool is not None:
            for pin_blocks in self._pins_issued:
                self._block_pool.free_blocks(pin_blocks)
        self._pins_issued, self._pins_staged = self._pins_staged, []

        self._absorb_block_allocations(scheduler_output)
        self._stage_filled_attention_blocks(scheduler_output)
        meta = HostTierMeta(
            restores=self._staged_restores,
            offloads=self._staged_offloads,
            zeros=self._staged_zeros,
        )
        self._staged_restores = {}
        self._staged_offloads = {}
        self._staged_zeros = {}
        self._last_step_write_slots.extend(
            s for ops in meta.offloads.values() for _, s in ops
        )
        return meta

    # Request lifecycle -------------------------------------------------

    def on_new_request(self, request: "Request") -> None:
        self._requests[request.request_id] = request

    def update_connector_output(self, connector_output) -> None:
        return

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        # Non-HMA entry point; HMA models use request_finished_all_groups.
        self._tracks.pop(request.request_id, None)
        self._requests.pop(request.request_id, None)
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Pin and save the tail boundary states; never hold via deferred free.

        The saved state must correspond EXACTLY to the aligned boundary
        ``B = k * block_size`` the index advertises: mamba state is
        cumulative and position-specific, and the live final block covers
        an unaligned ``num_computed_tokens`` (measured: nondeterministic
        garbled resumes proportional to the overhang). The engine's own
        align-mode prefix caching provides the snapshot we need - when a
        boundary is crossed, the state migrates into the next block and
        the boundary block is frozen, hashed (``cache_blocks``) and freed
        into the pool's prefix cache. Look that block up by the boundary
        hash and save its bytes.

        Blocks are protected with a plain reference (``BlockPool.touch``)
        and released once the copy's in-order-stream window has passed;
        the engine's HMA deferred-free path corrupts the block pool
        (differential-tested 2026-08-28), so this never holds.
        """
        track = self._tracks.pop(request.request_id, None)
        self._requests.pop(request.request_id, None)
        finished_owner = self._owner(request, track)
        del track
        bs = self.hash_block_size
        if not self.state_groups or self._block_pool is None:
            return False, None
        max_boundary = min(
            request.num_computed_tokens // bs, len(request.block_hashes)
        )
        if max_boundary <= 0:
            return False, None
        # Frozen states materialize only at positions that were a chunk-end
        # column at some scheduling step (this fork's align mode keeps ONE
        # live column per chunk; intermediate positions stay null and
        # cache_full_blocks skips them). Scan down from the last full block
        # to the deepest boundary every mamba group actually has cached; the
        # gap above it (at most one prefill chunk) re-prefills on resume.
        targets: list[int] = []
        boundary = 0
        scan_floor = max(1, max_boundary - 64)
        for j in range(max_boundary, scan_floor - 1, -1):
            cached = self._block_pool.get_cached_block(
                request.block_hashes[j - 1], self.state_groups
            )
            if cached:
                boundary = j
                targets = [blk.block_id for blk in cached]
                break
        if boundary <= 0:
            # Nothing cached in reach (short single-chunk request, or the
            # states were evicted): the tail is not resumable - skip.
            logger.debug(
                "host-tier: no cached boundary state for %s within blocks "
                "[%d, %d] (computed=%d); skipping tail save",
                request.request_id[-8:],
                scan_floor,
                max_boundary,
                request.num_computed_tokens,
            )
            return False, None
        slots = self.index.stage_tail_states(
            finished_owner,
            boundary,
            len(self.state_groups),
            boundary_hash=request.block_hashes[boundary - 1],
        )
        if slots is None:
            return False, None
        pin_blocks = [self._block_pool.blocks[b] for b in targets]
        self._block_pool.touch(pin_blocks)
        self._pins_staged.append(pin_blocks)
        ops = [
            (targets[tier_gid], slots[tier_gid])
            for tier_gid in range(len(self.state_groups))
        ]
        self._offload_seq += 1
        self._staged_offloads[self._offload_seq] = ops
        logger.info(
            "host-tier: pinned boundary-state save for %s at block %d",
            request.request_id[-8:],
            boundary,
        )
        return False, None

    # ==================================================================
    # Worker side
    # ==================================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        from vllm.v1.worker.gpu.kv_tier_dma import KVTierDMA

        # Find the packed slab: the storage shared by the most layers.
        # Layers outside it (if any) are not tier-managed; aiming the DMA at
        # the wrong allocation would corrupt unrelated GPU memory.
        by_storage: dict[int, list[str]] = {}
        storages = {}
        for name, t in kv_caches.items():
            st = t.untyped_storage()
            key = st.data_ptr()
            by_storage.setdefault(key, []).append(name)
            storages[key] = (st, t.device)
        packed_key = max(by_storage, key=lambda k: len(by_storage[k]))
        outside = {
            k: v for k, v in by_storage.items() if k != packed_key
        }
        if outside:
            logger.warning(
                "host-tier: %d layers live outside the packed slab and are "
                "NOT tier-managed: %s",
                sum(len(v) for v in outside.values()),
                [v[:2] for v in outside.values()],
            )
        storage, device = storages[packed_key]
        backing = torch.empty(0, dtype=torch.int8, device=device)
        backing.set_(storage)
        any_tensor = kv_caches[by_storage[packed_key][0]]
        if backing.numel() % self.block_stride:
            raise RuntimeError(
                "host-tier: packed KV backing is not a whole number of block "
                f"strides ({backing.numel()} % {self.block_stride})"
            )
        logger.info(
            "host-tier: packed slab shared by %d/%d layers, %d bytes",
            len(by_storage[packed_key]),
            len(kv_caches),
            backing.numel(),
        )
        self._dma = KVTierDMA(
            backing, self.block_stride, self.num_slots, any_tensor.device
        )
        logger.info(
            "host-tier: arena %d slots x %d bytes (%.1f GiB pinned) per rank",
            self.num_slots,
            self.block_stride,
            self.num_slots * self.block_stride / (1 << 30),
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, HostTierMeta) or self._dma is None:
            return
        from vllm.v1.worker.gpu.kv_tier_dma import TierOpBatch

        self._dma.fence_restores()
        # Everything is issued here rather than in wait_for_save:
        # start_load_kv runs on EVERY step, including empty (no_forward)
        # steps, while wait_for_save is skipped there - and a finished
        # request's tail save typically lands on exactly such a step. All
        # referenced blocks were written by prior steps, and the copy
        # stream waits on compute, so pre-forward issuance is safe.
        for seq, ops in meta.offloads.items():
            self._dma.issue(TierOpBatch(seq=seq, offload=ops, restore=[]))
        for req_id, ops in meta.restores.items():
            self._seq += 1
            self._pending_restore_reqs[self._seq] = req_id
            zero = meta.zeros.get(req_id, [])
            logger.info(
                "host-tier: issuing restore seq=%d req=%s ops=%d zero=%d",
                self._seq,
                req_id[-8:],
                len(ops),
                len(zero),
            )
            self._dma.issue(
                TierOpBatch(seq=self._seq, offload=[], restore=ops, zero=zero)
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata, **kwargs
    ) -> None:
        return

    def wait_for_save(self) -> None:
        # All issuance happens in start_load_kv (see note there).
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self._dma is None:
            return None, None
        done_recving: set[str] = set()
        for seq in self._dma.poll_done():
            req_id = self._pending_restore_reqs.pop(seq, None)
            if req_id is not None:
                done_recving.add(req_id)
        if done_recving:
            logger.info(
                "host-tier: worker done recv=%s",
                [r[-8:] for r in done_recving],
            )
        return None, done_recving or None

    def shutdown(self) -> None:
        if getattr(self, "_dma", None) is not None:
            self._dma.flush()
