# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host KV tier: pinned-RAM second tier behind the GPU block pool.

A KV connector that mirrors trajectories into a pinned host arena and
restores them when a request's prefix misses the GPU cache but matches a
stored trajectory.

Geometry. A logical POSITION is one scheduler block: the LCM of every cache
group's block size (256 tokens on DeepSeek-V4, whose compressor groups page
at 4-64 tokens under a 256-token MLA group). ``request.block_hashes`` is at
the GCD granularity the scheduler hashes at; the tier reads the hash that
closes each position. Each full-attention group contributes
``S // block_size`` sub-blocks per position, and a position's sub-blocks of
every such group share one host slot row, column-addressed. Both roles lay
the row out from ``KVCacheConfig.group_block_bytes``, which the planner
records from the real per-layer specs (the scheduler's copy of a uniform
group is flattened to one arbitrary member and cannot recompute it).

Contract (matches the engine's hybrid semantics, learned the hard way):
- Full-attention blocks are immutable once full: offloaded at fill time
  (one step behind the verified-token watermark so speculative rejections
  can never be captured), restored across the whole resumed span.
- The TAIL is whatever exists only at a boundary and is snapshotted once,
  when the request finishes, from the blocks the engine froze into its
  prefix cache: mamba align-mode state (one block per group at the
  boundary) and sliding-window attention (the in-window blocks, which the
  engine nulls and frees a few tokens later - only the frozen, hashed
  copies survive). A trajectory with a tail is resumable exactly at its
  recorded boundary - the natural shape of append-only agentic traffic.
- Attention-only models (GLM-5.2: MLA + DSA indexer) have no tail; a
  finished trajectory resumes at any gap-free prefix, and the per-position
  chain-hash comparison is the whole correctness check.
- Ring (CircularBufferSpec) groups are rolling per-request state, excluded
  from prefix caching by their own manager and reconstructed by
  allocate_external_computed_blocks; the tier ignores them (zeroes them on
  resume for determinism).
- Restores ride the framework's async-KV-load path
  (WAITING_FOR_REMOTE_KVS): issued while the request waits, hidden under
  running decode, and re-queried per scheduling step if clipped. Source
  slots are pinned in the index from plan to issue.

Config (kv_transfer_config.kv_connector_extra_config):
    host_tier_gb_per_rank: pinned arena size per rank in GiB (required).
"""

from __future__ import annotations

import logging
import math
import os
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
from vllm.v1.core.kv_cache_utils import _group_block_bytes
from vllm.v1.core.kv_tier_index import HostKVTierIndex
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu.kv_tier_dma import KVTierDMA

logger = init_logger(__name__)

# Diagnostic: VLLM_KV_TIER_IDLE=1 keeps the connector configured (so the
# packed-slab layout and registration are unchanged) but stages no offloads,
# serves no lookups and issues no DMA - for isolating tier activity in
# corruption bisections.

_TIER_IDLE = os.environ.get("VLLM_KV_TIER_IDLE", "0") == "1"
# VLLM_KV_TIER_DEBUG=1: this connector's debug logging (lookup misses with
# explain_miss, staging and sealing) without turning on DEBUG globally.
_TIER_DEBUG = os.environ.get("VLLM_KV_TIER_DEBUG", "0") == "1"


def _dbg(msg: str, *args) -> None:
    """Connector diagnostics: INFO under VLLM_KV_TIER_DEBUG=1 (vLLM's log
    handler drops DEBUG records regardless of this logger's level)."""
    if _TIER_DEBUG:
        logger.info(msg, *args)
    else:
        logger.debug(msg, *args)


# (slot, col, group, gpu_block): see vllm.v1.worker.gpu.kv_tier_dma.
TierOp = tuple[int, int, int, int]


@dataclass
class HostTierMeta(KVConnectorMetadata):
    # req_id -> restore ops for held requests.
    restores: dict[str, list[TierOp]] = field(default_factory=dict)
    # batch_seq -> offload ops.
    offloads: dict[int, list[TierOp]] = field(default_factory=dict)
    # req_id -> [(group, gpu_block), ...] blocks to zero alongside the
    # restore (the ring block, whose framework zeroing was skipped for the
    # async-load range but which the tier deliberately does not restore).
    zeros: dict[str, list[tuple[int, int]]] = field(default_factory=dict)


@dataclass
class _ReqTrack:
    # Per-config-group block ids, position-indexed at the GROUP's block
    # size; -1 marks null blocks.
    group_blocks: list[list[int]]
    staged_upto: int = 0  # positions
    planned_blocks: int = 0  # positions
    # planned_attn_slots[i][j]: host slot of sub-block j of position i.
    planned_attn_slots: list[list[int]] = field(default_factory=list)
    # planned_tail_slots[t]: slots of the t-th tail group's snapshot.
    planned_tail_slots: list[list[int]] = field(default_factory=list)
    planned_start: int = 0
    restored_upto: int = 0  # positions already staged for restore
    # Trajectory key this request stages into: the adopted trajectory's
    # owner when the request resumed from the tier (its true lineage), the
    # request's own id otherwise. Never derived from shared content.
    owner: str | None = None
    # Restore-source slots pinned in the index for this plan.
    pinned_slots: list[int] = field(default_factory=list)


class HostTierConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        tier_gb = float(extra.get("host_tier_gb_per_rank", 0))
        if tier_gb <= 0:
            raise ValueError(
                "HostTierConnector requires kv_connector_extra_config."
                "host_tier_gb_per_rank > 0"
            )

        groups = kv_cache_config.kv_cache_groups
        self.num_groups = len(groups)
        # Bytes of one block per group. The planner records this from the
        # real per-layer specs; the fallback recomputes from whatever specs
        # this role holds (exact on the worker; on the scheduler only for
        # non-uniform groups - the flattened uniform spec would understate a
        # heterogeneous group, the split brain that once planned 59811 slots
        # against a 15339-slot arena).
        gb = list(getattr(kv_cache_config, "group_block_bytes", None) or [])
        if len(gb) != len(groups):
            gb = _group_block_bytes(groups)
        self.group_bytes = gb

        # The worker holds UniformTypeKVCacheSpecs groups; the scheduler's
        # copy is flattened to one member. Classify and size by a member
        # either way (a uniform group's members share type and geometry).
        def rep(spec):
            if isinstance(spec, UniformTypeKVCacheSpecs):
                return next(iter(spec.kv_cache_specs.values()))
            return spec

        reps = [rep(g.kv_cache_spec) for g in groups]
        self.attn_groups: list[int] = []
        self.window_groups: list[int] = []
        self.state_groups: list[int] = []
        self.ring_groups: list[int] = []
        for gid, spec in enumerate(reps):
            if isinstance(spec, MambaSpec):
                self.state_groups.append(gid)
            elif isinstance(spec, CircularBufferSpec):
                self.ring_groups.append(gid)
            elif isinstance(spec, SlidingWindowSpec):
                self.window_groups.append(gid)
            elif isinstance(spec, AttentionSpec):
                self.attn_groups.append(gid)
            else:
                raise ValueError(
                    f"host tier: unsupported KV cache spec {type(spec).__name__}"
                    f" in group {gid}"
                )
        if not self.attn_groups:
            raise ValueError("host tier requires a full-attention group")

        # Token geometry, mirroring resolve_kv_cache_block_sizes: the
        # scheduler aligns at the LCM of the group block sizes and hashes at
        # the GCD of the prefix-cacheable ones (prefix_match_unit if set).
        self.group_block_size = [spec.block_size for spec in reps]
        dcp = getattr(
            getattr(vllm_config, "parallel_config", None),
            "decode_context_parallel_size",
            1,
        )
        assert dcp == 1, "host tier does not support decode context parallel"
        S = math.lcm(*self.group_block_size)
        cache_config = vllm_config.cache_config
        if len(groups) == 1 or any(
            self.group_block_size[gid] != cache_config.block_size
            for gid in self.state_groups
        ):
            H = S
        else:
            cacheable = [
                bs
                for spec, bs in zip(reps, self.group_block_size)
                if getattr(spec, "prefix_cacheable", True)
            ] or self.group_block_size
            unit = getattr(cache_config, "prefix_match_unit", None)
            H = unit if unit is not None else math.gcd(*cacheable)
        assert S % H == 0, (S, H)
        self.pos_tokens = S
        self.hash_block_size = H
        self.hashes_per_pos = S // H

        # Slot-row layout of one position: (group, col, sub-index) per
        # sub-block, full-attention groups in config order.
        self.sub_layout: list[tuple[int, int, int]] = []
        col = 0
        for gid in self.attn_groups:
            r = S // self.group_block_size[gid]
            for k in range(r):
                self.sub_layout.append((gid, col, k))
                col += gb[gid]
        self.pos_row_bytes = col
        # Tail groups snapshot one block per slot at column 0.
        self.tail_groups = self.state_groups + self.window_groups
        self.window_of = {gid: reps[gid].sliding_window for gid in self.window_groups}
        self.requires_tail = bool(self.tail_groups)
        self.slot_bytes = max(col, max(gb))
        self.num_slots = int(tier_gb * (1 << 30)) // self.slot_bytes
        if self.num_slots <= 0:
            raise ValueError("host tier smaller than one slot")

        for gid, spec in enumerate(reps):
            logger.info(
                "host-tier: group %d: %s x%d layers, block %d tokens, "
                "%d B/block, sliding_window=%s",
                gid,
                type(spec).__name__,
                len(groups[gid].layer_names),
                self.group_block_size[gid],
                gb[gid],
                getattr(spec, "sliding_window", None),
            )
        if role == KVConnectorRole.SCHEDULER:
            logger.info(
                "host-tier: %d slots x %d B, position %d tokens (%d sub-"
                "blocks, hashes at %d), attn groups %s, window groups %s, "
                "state groups %s, ring groups %s",
                self.num_slots,
                self.slot_bytes,
                S,
                len(self.sub_layout),
                H,
                self.attn_groups,
                self.window_groups,
                self.state_groups,
                self.ring_groups,
            )
            self.index = HostKVTierIndex(
                self.num_slots,
                requires_tail=self.requires_tail,
                slots_per_position=len(self.sub_layout),
            )
            self._tracks: dict[str, _ReqTrack] = {}
            self._requests: dict[str, Request] = {}
            self._staged_restores: dict[str, list[TierOp]] = {}
            self._staged_offloads: dict[int, list[TierOp]] = {}
            self._staged_zeros: dict[str, list[tuple[int, int]]] = {}
            # Writes issued last step; confirmed next build (in-order copy
            # stream: any restore issued later observes completed writes).
            self._last_step_write_slots: list[int] = []
            self._offload_seq = 0
            self._block_pool = None
            # Blocks ref-pinned for in-flight tail saves, released one meta
            # build after their copies were issued (same in-order-stream
            # window as slot confirmation). Two-phase: staged -> issued.
            self._pins_staged: list[list] = []
            self._pins_issued: list[list] = []
            # Restore-source slots pinned for planned restores, same window.
            self._slot_pins_staged: list[list[int]] = []
            self._slot_pins_issued: list[list[int]] = []
        else:
            self._dma: KVTierDMA | None = None
            self._pending_restore_reqs: dict[int, str] = {}
            self._seq = 1 << 20

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        self._block_pool = gpu_block_pool

    # ==================================================================
    # Geometry helpers
    # ==================================================================

    def _pos_hash(self, request: Request, pos: int):
        """Chain hash closing position `pos` (certifies the whole prefix)."""
        return request.block_hashes[(pos + 1) * self.hashes_per_pos - 1]

    def _pos_hashes(self, request: Request) -> list:
        n = len(request.block_hashes) // self.hashes_per_pos
        return [self._pos_hash(request, i) for i in range(n)]

    def _group_hash(self, request: Request, gid: int, m: int):
        """Chain hash closing block `m` of group `gid` (its own size)."""
        return request.block_hashes[
            (m + 1) * self.group_block_size[gid] // self.hash_block_size - 1
        ]

    def _num_full_positions(self, request: Request) -> int:
        return min(
            request.num_computed_tokens // self.pos_tokens,
            len(request.block_hashes) // self.hashes_per_pos,
        )

    def _tail_block_range(self, gid: int, tokens: int) -> tuple[int, int]:
        """Group-block index range a tail group snapshots at `tokens`.

        Mamba: the single boundary block. Sliding window: the in-window
        blocks, mirroring SlidingWindowManager.get_num_skipped_tokens and
        the null padding of add_local_computed_blocks.
        """
        b = self.group_block_size[gid]
        last = tokens // b
        if gid in self.window_of:
            skipped = max(0, tokens - self.window_of[gid] + 1) // b
            return skipped, last
        return last - 1, last

    # ==================================================================
    # Scheduler side
    # ==================================================================

    @staticmethod
    def _owner(request: Request, track: _ReqTrack | None) -> str:
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
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if _TIER_IDLE:
            return 0, False
        S = self.pos_tokens
        track = self._tracks.get(request.request_id)
        if track is not None and track.planned_blocks:
            # Re-query mid progressive restore: report the remainder.
            remaining = track.planned_blocks * S - num_computed_tokens
            if remaining > 0:
                return remaining, True
        hashes = self._pos_hashes(request)
        hit = self.index.lookup(hashes)
        if hit is None:
            if (_TIER_DEBUG or logger.isEnabledFor(logging.DEBUG)) and len(hashes) >= 8:
                _dbg(
                    "host-tier: lookup miss for %s (%d positions, computed %d): %s",
                    request.request_id[-8:],
                    len(hashes),
                    num_computed_tokens,
                    self.index.explain_miss(hashes),
                )
            return 0, False
        hit_owner, n_blocks, attn_slots, tail_slots = hit
        n_tokens = n_blocks * S
        # With a tail, resume is only possible exactly at the stored
        # boundary; either way at least one token must remain to compute.
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
        track.planned_tail_slots = tail_slots
        track.planned_start = num_computed_tokens // S
        track.restored_upto = num_computed_tokens // S
        # Pin every source slot until the copies are issued: nothing else
        # keeps another request's staging from reclaiming this trajectory
        # and re-staging the slots before the DMA reads them.
        pinned = [s for row in attn_slots for s in row] + [
            s for g in tail_slots for s in g
        ]
        self.index.pin_slots(pinned)
        track.pinned_slots = pinned
        logger.info(
            "host-tier: hit for %s: resume at position %d (%d tokens)",
            request.request_id[-8:],
            n_blocks,
            n_tokens,
        )
        return n_tokens - num_computed_tokens, True

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
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
        S = self.pos_tokens
        # The scheduler may clip the external span to its budget and
        # re-query per step; stage restores for the newly covered
        # positions. Progress is planned-relative: request.num_computed_
        # tokens is not advanced while the request waits on the async load.
        covered = min(
            track.restored_upto + num_external_tokens // S,
            track.planned_blocks,
        )
        ops: list[TierOp] = []
        for logical in range(track.restored_upto, covered):
            row = track.planned_attn_slots[logical]
            for j, (gid, col, k) in enumerate(self.sub_layout):
                gb = track.group_blocks[gid]
                idx = logical * (S // self.group_block_size[gid]) + k
                if idx >= len(gb) or gb[idx] < 0:
                    logger.warning(
                        "host-tier: no target block for %s group %d block %d",
                        request.request_id[-8:],
                        gid,
                        idx,
                    )
                    self._abandon_plan(track)
                    return
                ops.append((row[j], col, gid, gb[idx]))
        zero_blocks: list[tuple[int, int]] = []
        if covered == track.planned_blocks:
            # The tail rides the FINAL restore chunk. Each tail group's
            # snapshot lands on the blocks the engine allocated for the
            # boundary: the mamba position-(k-1) block ([null] * (k-1) +
            # [real], matching the worker's state_idx seed) or the sliding
            # window's trailing real blocks after its null padding.
            tokens = track.planned_blocks * S
            for t, gid in enumerate(self.tail_groups):
                if t >= len(track.planned_tail_slots):
                    break
                slots = track.planned_tail_slots[t]
                gb = track.group_blocks[gid]
                first, last = self._tail_block_range(gid, tokens)
                targets = gb[first:last] if last <= len(gb) else []
                if len(targets) != len(slots) or any(b < 0 for b in targets):
                    logger.warning(
                        "host-tier: tail target mismatch for %s group %d: "
                        "need %d real blocks in [%d, %d), have %s",
                        request.request_id[-8:],
                        gid,
                        len(slots),
                        first,
                        last,
                        targets,
                    )
                    self._abandon_plan(track)
                    return
                ops.extend((slot, 0, gid, block) for slot, block in zip(slots, targets))
            # Zero the ring defensively (see module docstring: internal
            # hits run on a stale-claimed, never-zeroed ring; zero is the
            # same or strictly cleaner).
            for gid in self.ring_groups:
                zero_blocks.extend((gid, b) for b in track.group_blocks[gid] if b > 0)
        if ops:
            self._staged_restores.setdefault(request.request_id, []).extend(ops)
            if zero_blocks:
                self._staged_zeros.setdefault(request.request_id, []).extend(
                    zero_blocks
                )
            track.restored_upto = covered
            logger.info(
                "host-tier: staged %d restore ops for %s (positions %d..%d of %d)",
                len(ops),
                request.request_id[-8:],
                track.planned_start,
                covered,
                track.planned_blocks,
            )
        if covered == track.planned_blocks:
            track.planned_blocks = 0  # restore fully staged
            # The copies are issued with the next meta build; the pins ride
            # the same two-phase window as the block pins.
            self._slot_pins_staged.append(track.pinned_slots)
            track.pinned_slots = []

    def _abandon_plan(self, track: _ReqTrack) -> None:
        """Drop a planned restore that cannot be staged; nothing was issued
        for it, so its pins can be released immediately."""
        track.planned_blocks = 0
        self.index.unpin_slots(track.pinned_slots)
        track.pinned_slots = []

    # Offload staging ---------------------------------------------------

    def _absorb_block_allocations(self, scheduler_output: SchedulerOutput) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            track = self._tracks.get(new_req.req_id)
            if track is None:
                track = _ReqTrack(group_blocks=[[] for _ in range(self.num_groups)])
                self._tracks[new_req.req_id] = track
            for gid, ids in enumerate(new_req.block_ids):
                if gid < len(track.group_blocks) and len(ids) > len(
                    track.group_blocks[gid]
                ):
                    track.group_blocks[gid] = list(ids)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_ids is None:
                continue
            track = self._tracks.get(req_id)
            if track is None:
                continue
            for gid, group_new in enumerate(new_ids):
                if gid < len(track.group_blocks) and group_new:
                    track.group_blocks[gid].extend(group_new)

    def _stage_filled_attention_blocks(self, scheduler_output: SchedulerOutput) -> None:
        if _TIER_IDLE:
            return
        S = self.pos_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            track = self._tracks.get(req_id)
            request = self._requests.get(req_id)
            if track is None or request is None:
                continue
            n_full = min(
                self._num_full_positions(request),
                min(
                    (
                        len(track.group_blocks[g]) // (S // self.group_block_size[g])
                        for g in self.attn_groups
                    ),
                    default=0,
                ),
            )
            if track.staged_upto > n_full:
                track.staged_upto = n_full  # preemption reset
                continue
            owner = self._owner(request, track)
            for logical in range(track.staged_upto, n_full):
                block_hash = self._pos_hash(request, logical)
                ops: list[TierOp] = []
                for j, (gid, col, k) in enumerate(self.sub_layout):
                    idx = logical * (S // self.group_block_size[gid]) + k
                    block_id = track.group_blocks[gid][idx]
                    if block_id < 0:
                        continue
                    slot = self.index.stage_attention(owner, logical, block_hash, sub=j)
                    if slot is not None:
                        ops.append((slot, col, gid, block_id))
                if ops:
                    self._offload_seq += 1
                    self._staged_offloads[self._offload_seq] = ops
            if n_full > track.staged_upto:
                _dbg(
                    "host-tier: staged positions %d..%d of %s "
                    "(%d full, %d group0 blocks)",
                    track.staged_upto,
                    n_full,
                    req_id[-8:],
                    self._num_full_positions(request),
                    len(track.group_blocks[self.attn_groups[0]])
                    if self.attn_groups
                    else -1,
                )
            track.staged_upto = n_full

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        confirmed, self._last_step_write_slots = self._last_step_write_slots, []
        self.index.confirm_writes(confirmed)
        if self._pins_issued and self._block_pool is not None:
            for pin_blocks in self._pins_issued:
                self._block_pool.free_blocks(pin_blocks)
        self._pins_issued, self._pins_staged = self._pins_staged, []
        for pinned in self._slot_pins_issued:
            self.index.unpin_slots(pinned)
        self._slot_pins_issued, self._slot_pins_staged = self._slot_pins_staged, []

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
            op[0] for ops in meta.offloads.values() for op in ops
        )
        return meta

    # Request lifecycle -------------------------------------------------

    def on_new_request(self, request: Request) -> None:
        self._requests[request.request_id] = request

    def update_connector_output(self, connector_output) -> None:
        return

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        # Non-HMA entry point; this connector is SupportsHMA, so the
        # scheduler always calls request_finished_all_groups.
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Seal the trajectory; pin and save the tail snapshot if any.

        A saved tail must correspond EXACTLY to the aligned boundary
        ``B = k * S`` the index advertises: mamba state is cumulative and
        position-specific, and a sliding window's blocks are nulled as it
        slides, while the live final block covers an unaligned
        ``num_computed_tokens`` (measured: nondeterministic garbled resumes
        proportional to the overhang). The engine's own prefix caching
        provides the snapshot we need - when a boundary is crossed, the
        boundary blocks are frozen, hashed (``cache_blocks``) and freed
        into the pool's prefix cache. Look them up by hash and save them.

        Blocks are protected with a plain reference (``BlockPool.touch``)
        and released once the copy's in-order-stream window has passed;
        the engine's HMA deferred-free path corrupts the block pool
        (differential-tested 2026-08-28), so this never holds.
        """
        track = self._tracks.pop(request.request_id, None)
        self._requests.pop(request.request_id, None)
        finished_owner = self._owner(request, track)
        if track is not None and track.pinned_slots:
            # Finished before its restore was fully staged (aborted or
            # preempted mid-plan): release what was never issued.
            self.index.unpin_slots(track.pinned_slots)
        del track
        if not self.requires_tail:
            # Attention-only: nothing to snapshot, but the trajectory must
            # be sealed so it becomes matchable.
            if self.index.mark_complete(finished_owner):
                _dbg(
                    "host-tier: sealed attention-only trajectory for %s (owner %s)",
                    request.request_id[-8:],
                    finished_owner[-12:],
                )
            else:
                _dbg(
                    "host-tier: NOT sealed for %s (owner %s): %s",
                    request.request_id[-8:],
                    finished_owner[-12:],
                    self.index.explain_owner(finished_owner)
                    if hasattr(self.index, "explain_owner")
                    else "no trajectory / pending writes",
                )
            return False, None
        if self._block_pool is None:
            return False, None
        max_boundary = self._num_full_positions(request)
        if max_boundary <= 0:
            return False, None
        # Frozen blocks materialize only at positions that were a chunk-end
        # column at some scheduling step (this fork's align mode keeps ONE
        # live mamba column per chunk; intermediate positions stay null and
        # cache_full_blocks skips them). Scan down from the last full
        # position to the deepest boundary every tail group actually has
        # cached; the gap above it (at most one prefill chunk) re-prefills
        # on resume.
        S = self.pos_tokens
        targets: list[list[int]] = []
        boundary = 0
        scan_floor = max(1, max_boundary - 64)
        for j in range(max_boundary, scan_floor - 1, -1):
            tokens = j * S
            found: list[list[int]] = []
            for gid in self.tail_groups:
                first, last = self._tail_block_range(gid, tokens)
                ids: list[int] = []
                for m in range(first, last):
                    cached = self._block_pool.get_cached_block(
                        self._group_hash(request, gid, m), [gid]
                    )
                    if not cached:
                        break
                    ids.append(cached[0].block_id)
                if len(ids) != last - first:
                    found = []
                    break
                found.append(ids)
            if found:
                boundary = j
                targets = found
                break
        if boundary <= 0:
            # Nothing cached in reach (short single-chunk request, or the
            # blocks were evicted): the tail is not resumable - skip.
            logger.debug(
                "host-tier: no cached boundary blocks for %s within positions "
                "[%d, %d] (computed=%d); skipping tail save",
                request.request_id[-8:],
                scan_floor,
                max_boundary,
                request.num_computed_tokens,
            )
            return False, None
        slots = self.index.stage_tail(
            finished_owner,
            boundary,
            [len(t) for t in targets],
            boundary_hash=self._pos_hash(request, boundary - 1),
        )
        if slots is None:
            return False, None
        pin_blocks = [self._block_pool.blocks[b] for ids in targets for b in ids]
        self._block_pool.touch(pin_blocks)
        self._pins_staged.append(pin_blocks)
        ops: list[TierOp] = []
        for gid, ids, group_slots in zip(self.tail_groups, targets, slots):
            ops.extend((slot, 0, gid, b) for slot, b in zip(group_slots, ids))
        self._offload_seq += 1
        self._staged_offloads[self._offload_seq] = ops
        logger.info(
            "host-tier: pinned tail save for %s at position %d (%d blocks)",
            request.request_id[-8:],
            boundary,
            len(ops),
        )
        return False, None

    # ==================================================================
    # Worker side
    # ==================================================================

    def _layer_page_bytes(self) -> dict[str, int]:
        """Per-layer page size from the group specs (unpacked layouts)."""
        out: dict[str, int] = {}
        for group in self._kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            for name in group.layer_names:
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    out[name] = spec.kv_cache_specs[name].page_size_bytes
                else:
                    out[name] = spec.page_size_bytes
        return out

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Build per-group DMA segments from the planner's tensor layout.

        The layout is driven by kv_cache_config.kv_cache_tensors - what the
        allocator actually followed - never inferred from the tensors: a
        storage-pointer vote would silently leave any layer outside the
        winning allocation unmanaged, and a per-layer layout would then
        restore only part of a block (wrong KV cached under a real hash).
        Every tier-managed layer must be accounted for, or we refuse.
        """
        from vllm.v1.worker.gpu.kv_tier_dma import KVTierDMA, TierSegment

        def flat(t: torch.Tensor) -> torch.Tensor:
            """The whole allocation behind a layer view, as flat int8."""
            b = torch.empty(0, dtype=torch.int8, device=t.device)
            b.set_(t.untyped_storage())
            return b

        cfg = self._kv_cache_config
        groups = cfg.kv_cache_groups
        packed_layers = {
            ln: t for t in cfg.kv_cache_tensors if t.block_stride for ln in t.shared_by
        }
        unpacked_layers = {
            ln: t
            for t in cfg.kv_cache_tensors
            if not t.block_stride
            for ln in t.shared_by
        }
        strides = {t.block_stride for t in packed_layers.values()}
        assert len(strides) <= 1, f"host-tier: mixed packed strides {strides}"
        device = next(iter(kv_caches.values())).device
        page = self._layer_page_bytes()

        slab_rows: torch.Tensor | None = None
        if packed_layers:
            # One allocation; every packed tensor aliases it. A group's
            # layers occupy [0, group_bytes) of each block-stride row.
            stride = strides.pop()
            first = next(ln for ln in packed_layers if ln in kv_caches)
            backing = flat(kv_caches[first])
            if backing.numel() % stride:
                raise RuntimeError(
                    "host-tier: packed KV backing is not a whole number of "
                    f"block strides ({backing.numel()} % {stride})"
                )
            slab_rows = backing.view(-1, stride)

        group_segments: dict[int, list[TierSegment]] = {}
        n_packed = n_layers = 0
        for gid, group in enumerate(groups):
            names = [ln for ln in group.layer_names if ln in kv_caches]
            if not names:
                continue
            n_layers += len(names)
            if all(ln in packed_layers for ln in names):
                assert slab_rows is not None
                group_segments[gid] = [
                    TierSegment(slab_rows[:, : self.group_bytes[gid]])
                ]
                n_packed += len(names)
                continue
            segs: list[TierSegment] = []
            for ln in group.layer_names:
                if ln not in kv_caches:
                    continue
                t = unpacked_layers.get(ln)
                if t is None:
                    raise RuntimeError(
                        f"host-tier: layer {ln} has no KV tensor in the planner layout"
                    )
                if len(t.shared_by) != 1:
                    raise RuntimeError(
                        "host-tier: KV tensor shared by several layers "
                        f"{t.shared_by[:3]}... is not block-addressable by "
                        "one id (upstream group-pool layout); use the "
                        "packed slab (enable_cross_layers_blocks) instead"
                    )
                segs.append(TierSegment.from_flat(flat(kv_caches[ln]), page[ln]))
            width = sum(s.width for s in segs)
            if width != self.group_bytes[gid]:
                raise RuntimeError(
                    f"host-tier: group {gid} segments total {width} B but the "
                    f"planner says {self.group_bytes[gid]} B per block"
                )
            group_segments[gid] = segs
        missing = set(kv_caches) - {ln for g in groups for ln in g.layer_names}
        if missing:
            raise RuntimeError(
                f"host-tier: {len(missing)} layers are in no cache group: "
                f"{sorted(missing)[:4]}..."
            )
        logger.info(
            "host-tier: %d/%d layers in the packed slab, %d groups, slot %d B",
            n_packed,
            n_layers,
            len(group_segments),
            self.slot_bytes,
        )
        self._dma = KVTierDMA(group_segments, self.slot_bytes, self.num_slots, device)
        logger.info(
            "host-tier: arena %d slots x %d bytes (%.1f GiB pinned) per rank",
            self.num_slots,
            self.slot_bytes,
            self.num_slots * self.slot_bytes / (1 << 30),
        )

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
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
        dma = getattr(self, "_dma", None)
        if dma is not None:
            dma.flush()
