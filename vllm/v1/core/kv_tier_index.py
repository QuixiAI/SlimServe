# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side index for the host KV tier (trajectory-centric).

The host tier mirrors a trajectory's immutable full-attention blocks at
fill time and, on models that have them, its boundary snapshot ("tail")
when the request finishes: mamba/GDN state blocks and sliding-window
attention windows, neither of which survives at earlier positions.

A logical POSITION is one scheduler block (the LCM of every cache group's
block size). A position costs `slots_per_position` host slots: one per
sub-block of each full-attention group (a group with a smaller block size
contributes several sub-blocks per position). A position counts only when
every sub-slot is held; the resumable span ends at the first incomplete
position.

Resumability depends on whether the model carries a tail:

- With state or window groups, a trajectory is resumable only at its
  recorded tail boundary: mamba align-mode state exists only there, and a
  sliding window's out-of-window blocks are nulled by the engine.
- Attention-only models (MLA/DSA: GLM-5.2, ...) have nothing to pair, so
  any gap-free staged prefix of a COMPLETED trajectory resumes. Completion
  is marked explicitly at finish (mark_complete) because a trajectory its
  owner is still extending must not be adopted by another request.

Lookups match a new request's hash chain against a stored trajectory: a hit
restores attention positions [0, n) plus any tail blocks, and the request
resumes computing from there.

Ownership: a trajectory is keyed by the request id that CREATED it, and a
request that resumes from a trajectory adopts that key so the conversation's
next turn extends the same lineage. Keys must never be derived from shared
content (an earlier scheme keyed on the first block hash, which collapsed
every conversation sharing a system prompt into one chimera trajectory that
matched nobody - 6 hits per 197 saves in production). Concurrent requests
that share a prefix therefore build separate trajectories; the duplication
is reclaimed by LRU, while correctness is guarded by full-chain hash
comparison at lookup plus the tail_hash check in resumable_blocks.

Eviction is trajectory-affine and LRU: reclamation frees whole cold
trajectories, never individual slots. Slots a planned restore will read are
pinned (refcounted) until the copies are issued.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import cast

from vllm.v1.core.kv_cache_utils import BlockHash

_EMPTY_HASH = BlockHash(b"")


@dataclass
class Trajectory:
    hashes: list[BlockHash] = field(default_factory=list)
    # attn_slots[i][j]: host slot holding sub-block j of position i. All of
    # a position's sub-blocks share ONE slot row (column-addressed), so the
    # entries of a row are the same slot; an entry is None until that
    # sub-block is staged, and the resumable span ends at the first
    # position with any hole.
    attn_slots: list[list[int | None]] = field(default_factory=list)
    # Tail snapshot recorded at position `tail_boundary`: tail_slots[t] is
    # the slot list of the t-th tail group (one slot per mamba state block,
    # one per in-window block of a sliding-window group).
    tail_boundary: int = -1
    tail_slots: list[list[int]] = field(default_factory=list)
    # Chain hash at the tail boundary (position tail_boundary - 1) recorded
    # by the request that saved the tail. Resumability requires the staged
    # attention chain to carry the SAME hash there: a tail paired with
    # another conversation's attention blocks would resume with the wrong
    # state (silent output corruption), so a mismatched trajectory is
    # simply dead rather than dangerously matchable.
    tail_hash: BlockHash = _EMPTY_HASH
    tail_pending: bool = False  # tail writes still in flight
    # Attention-only models have no tail to record, so completion is marked
    # explicitly when the owning request finishes (see mark_complete).
    complete: bool = False
    last_touch: float = 0.0

    def _gap_free_prefix(self, limit: int) -> int:
        n = 0
        for row in self.attn_slots[:limit]:
            if any(s is None for s in row):
                break
            n += 1
        return n

    def resumable_blocks(self, requires_tail: bool = True) -> int:
        """Longest resumable gap-free attention prefix, in positions."""
        if not requires_tail:
            if not self.complete:
                return 0
            return self._gap_free_prefix(len(self.attn_slots))
        if self.tail_boundary <= 0 or self.tail_pending:
            return 0
        if (
            not self.tail_hash
            or self.tail_boundary > len(self.hashes)
            or self.hashes[self.tail_boundary - 1] != self.tail_hash
        ):
            return 0
        n = self._gap_free_prefix(self.tail_boundary)
        return n if n == self.tail_boundary else 0

    def resumable_span(self, requires_tail: bool) -> int:
        """Positions the resumable prefix is drawn from (diagnostics)."""
        return self.tail_boundary if requires_tail else len(self.attn_slots)

    def all_tail_slots(self) -> list[int]:
        return [s for slots in self.tail_slots for s in slots]


class HostKVTierIndex:
    """Trajectory-centric host tier placement and lookup."""

    def __init__(
        self,
        num_slots: int,
        requires_tail: bool = True,
        slots_per_position: int = 1,
    ):
        assert num_slots > 0
        assert slots_per_position >= 1
        self.num_slots = num_slots
        self.requires_tail = requires_tail
        self.slots_per_position = slots_per_position
        self._free: list[int] = list(range(num_slots - 1, -1, -1))
        self._trajectories: OrderedDict[str, Trajectory] = OrderedDict()
        # slot -> number of writes in flight (a position's sub-blocks may
        # be staged in different steps into the same slot).
        self._pending_write: dict[int, int] = {}
        # Host slots a planned restore will read, refcounted. lookup() hands
        # out slot ids, and the DMA reads them a step or more later; without
        # a pin, another request's staging could reclaim the source
        # trajectory in between and re-stage those very slots - and the
        # worker issues a batch's offloads BEFORE its restores, so the
        # overwrite would land first (silent corruption, then cached under
        # the requester's hashes). Pinned slots are never reclaimed.
        self._restore_pins: dict[int, int] = {}

    # ------------------------------------------------------------------ write

    def _alloc_slot(self, protect: str) -> int | None:
        if not self._free and not self._reclaim(protect):
            return None
        slot = self._free.pop()
        self._mark_pending(slot)
        return slot

    def _mark_pending(self, slot: int) -> None:
        self._pending_write[slot] = self._pending_write.get(slot, 0) + 1

    def _release(self, slots: list[int]) -> None:
        for s in dict.fromkeys(slots):
            self._pending_write.pop(s, None)
            self._free.append(s)

    def stage_attention(
        self, owner: str, logical: int, block_hash: BlockHash, sub: int = 0
    ) -> int | None:
        """Reserve a slot for sub-block `sub` of position `logical`."""
        assert 0 <= sub < self.slots_per_position
        traj = self._trajectories.setdefault(owner, Trajectory())
        self.touch(owner)
        while len(traj.attn_slots) <= logical:
            traj.attn_slots.append([None] * self.slots_per_position)
            traj.hashes.append(_EMPTY_HASH)
        row = traj.attn_slots[logical]
        if row[sub] is not None:
            return None
        held = next((s for s in row if s is not None), None)
        if held is not None:
            slot = held
            self._mark_pending(slot)
        else:
            fresh = self._alloc_slot(owner)
            if fresh is None:
                return None
            slot = fresh
        row[sub] = slot
        traj.hashes[logical] = block_hash
        # The trajectory is growing again (a resumed request extending an
        # adopted lineage), so it is no longer a sealed, matchable snapshot
        # until its owner finishes. Re-staging a held position returns
        # above without reaching here, so a pure resume does not unseal.
        traj.complete = False
        return slot

    def stage_tail(
        self,
        owner: str,
        boundary: int,
        counts: list[int],
        boundary_hash: BlockHash = _EMPTY_HASH,
    ) -> list[list[int]] | None:
        """Reserve slots for the tail snapshot of `owner` at `boundary`.

        `counts[t]` is how many blocks the t-th tail group snapshots (1 for
        a mamba state, the in-window block count for a sliding window).
        `boundary_hash` is the saver's chain hash at position boundary - 1;
        resumability later requires the staged attention chain to carry the
        same hash there (see Trajectory.tail_hash).

        Returns the slot lists or None when capacity is unavailable.
        Replaces any previously recorded tail (a trajectory grows; its old
        tail is superseded).
        """
        traj = self._trajectories.setdefault(owner, Trajectory())
        self.touch(owner)
        slots: list[list[int]] = []
        for n in counts:
            group: list[int] = []
            for _ in range(n):
                slot = self._alloc_slot(owner)
                if slot is None:
                    self._release([s for g in slots for s in g] + group)
                    return None
                group.append(slot)
            slots.append(group)
        self._release(traj.all_tail_slots())
        traj.tail_slots = slots
        traj.tail_boundary = boundary
        traj.tail_hash = boundary_hash
        traj.tail_pending = True
        return slots

    def pin_slots(self, slots: list[int]) -> None:
        """Protect planned restore sources from reclamation (refcounted)."""
        for s in slots:
            self._restore_pins[s] = self._restore_pins.get(s, 0) + 1

    def unpin_slots(self, slots: list[int]) -> None:
        for s in slots:
            n = self._restore_pins.get(s, 0) - 1
            if n > 0:
                self._restore_pins[s] = n
            else:
                self._restore_pins.pop(s, None)

    def mark_complete(self, owner: str) -> bool:
        """Mark `owner`'s trajectory finished and therefore matchable
        (attention-only models). Returns False when nothing was staged."""
        traj = self._trajectories.get(owner)
        if traj is None:
            return False
        traj.complete = True
        self.touch(owner)
        return True

    def confirm_writes(self, slots: list[int]) -> None:
        for slot in slots:
            n = self._pending_write.get(slot, 0) - 1
            if n > 0:
                self._pending_write[slot] = n
            else:
                self._pending_write.pop(slot, None)
        for traj in self._trajectories.values():
            if traj.tail_pending and not any(
                s in self._pending_write for s in traj.all_tail_slots()
            ):
                traj.tail_pending = False

    # ----------------------------------------------------------------- lookup

    def lookup(
        self, hashes: list[BlockHash]
    ) -> tuple[str, int, list[list[int]], list[list[int]]] | None:
        """Match `hashes` (one per position) against stored trajectories.

        Returns (owner, num_positions, attention_slots, tail_slots) for the
        deepest resumable trajectory whose hash prefix matches, or None.
        attention_slots[i][j] is the host slot of sub-block j of position i;
        the list is positional and never has holes. The owner lets a
        resuming request ADOPT the trajectory and extend it in place.
        """
        best: tuple[str, int, list[list[int]], list[list[int]]] | None = None
        best_owner: str | None = None
        for owner, traj in list(self._trajectories.items()):
            n = traj.resumable_blocks(self.requires_tail)
            if n <= 0:
                continue
            if n > len(hashes):
                # A trajectory longer than the query: with a tail the resume
                # point must be exactly the stored boundary, so it cannot
                # serve a shorter prefix. Attention-only trajectories are
                # resumable at any position, so the query's own length is
                # the match (a conversation continued from its earlier,
                # longer trajectory - the GLM-5.2/MI300X acceptance miss,
                # 2026-09-03: 141 staged positions vs 139 queried).
                if self.requires_tail:
                    continue
                n = len(hashes)
            if best is not None and n <= best[1]:
                continue
            if any(
                s in self._pending_write for row in traj.attn_slots[:n] for s in row
            ):
                continue
            if traj.hashes[:n] != hashes[:n]:
                continue
            rows = [list(row) for row in traj.attn_slots[:n]]
            # resumable_blocks guarantees a gap-free span; the consumer
            # indexes these rows positionally, so a hole would shift every
            # later restore onto the wrong block.
            assert all(s is not None for row in rows for s in row)
            tail = [list(g) for g in traj.tail_slots] if self.requires_tail else []
            best = (owner, n, cast(list[list[int]], rows), tail)
            best_owner = owner
        if best_owner is not None:
            self.touch(best_owner)
        return best

    def explain_miss(self, hashes: list[BlockHash]) -> str:
        """Diagnose why lookup(hashes) found nothing (debug aid)."""
        if not hashes:
            return "no hashes"
        notes = []
        for owner, traj in self._trajectories.items():
            if not traj.hashes or traj.hashes[0] != hashes[0]:
                continue
            span = traj.resumable_span(self.requires_tail)
            gap = next(
                (
                    i
                    for i, row in enumerate(traj.attn_slots[:span])
                    if any(s is None for s in row)
                ),
                None,
            )
            mism = next(
                (
                    i
                    for i in range(min(len(traj.hashes), len(hashes)))
                    if traj.hashes[i] != hashes[i]
                ),
                None,
            )
            pend = [
                s
                for row in traj.attn_slots[:span]
                for s in row
                if s in self._pending_write
            ]
            gate = (
                f"tail={traj.tail_boundary} tail_pending={traj.tail_pending}"
                if self.requires_tail
                else f"complete={traj.complete}"
            )
            notes.append(
                f"owner={owner[:12]} {gate} attn_len={len(traj.attn_slots)} "
                f"first_gap={gap} hash_mismatch_at={mism} "
                f"pending_attn={len(pend)} req_hashes={len(hashes)}"
            )
        return "; ".join(notes) or "no trajectory shares hashes[0]"

    def touch(self, owner: str) -> None:
        traj = self._trajectories.get(owner)
        if traj is not None:
            traj.last_touch = time.monotonic()
            self._trajectories.move_to_end(owner)

    # --------------------------------------------------------------- eviction

    def _traj_slots(self, traj: Trajectory) -> list[int]:
        """Distinct slots a trajectory holds (a row repeats its one slot)."""
        return list(
            dict.fromkeys(
                [s for row in traj.attn_slots for s in row if s is not None]
                + traj.all_tail_slots()
            )
        )

    def _busy(self, slots: list[int]) -> bool:
        """Slots with a write in flight or a planned restore reading them."""
        return any(s in self._pending_write or s in self._restore_pins for s in slots)

    def _reclaim(self, protect: str) -> bool:
        for owner in list(self._trajectories.keys()):
            if owner == protect:
                continue
            traj = self._trajectories[owner]
            slots = self._traj_slots(traj)
            if self._busy(slots):
                continue
            self._free.extend(slots)
            del self._trajectories[owner]
            return True
        return False

    def drop_owner(self, owner: str) -> None:
        traj = self._trajectories.get(owner)
        if traj is None:
            return
        slots = self._traj_slots(traj)
        if self._busy(slots):
            return
        self._free.extend(slots)
        del self._trajectories[owner]

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict[str, int]:
        return {
            "slots": self.num_slots,
            "used": self.num_slots - len(self._free),
            "trajectories": len(self._trajectories),
            "resumable": sum(
                1
                for t in self._trajectories.values()
                if t.resumable_blocks(self.requires_tail) > 0
            ),
            "pending_writes": len(self._pending_write),
            "pinned": len(self._restore_pins),
        }
