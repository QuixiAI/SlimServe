# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side index for the host KV tier (trajectory-centric).

The host tier mirrors a trajectory's immutable full-attention blocks at
fill time and its mamba/state tail-boundary blocks when the request
finishes. A trajectory is resumable only at its recorded tail boundary:
mamba (align-mode) state exists only there, mirroring the engine's own
resident-state semantics (earlier boundary positions are nulls).

Lookups match a new request's hash chain against a stored trajectory: a hit
restores attention blocks [0, tail) plus the tail state blocks, and the
request resumes computing from the tail boundary.

Eviction is trajectory-affine and LRU: reclamation frees whole cold
trajectories, never individual slots.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class Trajectory:
    hashes: list[BlockHash] = field(default_factory=list)
    # Attention slots, position-indexed: attn_slots[i] holds block i.
    # A position may be missing (None) if staging failed; the resumable
    # span ends at the first gap.
    attn_slots: list[int | None] = field(default_factory=list)
    # Tail-boundary state: logical block index -> {tier_state_gid: slot}.
    tail_boundary: int = -1
    tail_state_slots: dict[int, int] = field(default_factory=dict)
    tail_pending: bool = False  # tail-state writes still in flight
    last_touch: float = 0.0

    def resumable_blocks(self) -> int:
        """Longest gap-free attention prefix ending at the tail boundary."""
        if self.tail_boundary <= 0 or self.tail_pending:
            return 0
        n = 0
        for slot in self.attn_slots[: self.tail_boundary]:
            if slot is None:
                break
            n += 1
        return n if n == self.tail_boundary else 0


class HostKVTierIndex:
    """Trajectory-centric host tier placement and lookup."""

    def __init__(self, num_slots: int):
        assert num_slots > 0
        self.num_slots = num_slots
        self._free: list[int] = list(range(num_slots - 1, -1, -1))
        self._trajectories: OrderedDict[str, Trajectory] = OrderedDict()
        self._pending_write: set[int] = set()

    # ------------------------------------------------------------------ write

    def _alloc_slot(self, protect: str) -> int | None:
        if not self._free and not self._reclaim(protect):
            return None
        slot = self._free.pop()
        self._pending_write.add(slot)
        return slot

    def stage_attention(
        self, owner: str, logical: int, block_hash: BlockHash
    ) -> int | None:
        """Reserve a slot for attention block `logical` of `owner`."""
        traj = self._trajectories.setdefault(owner, Trajectory())
        self.touch(owner)
        while len(traj.attn_slots) <= logical:
            traj.attn_slots.append(None)
            traj.hashes.append(b"")
        if traj.attn_slots[logical] is not None:
            return None
        slot = self._alloc_slot(owner)
        if slot is None:
            return None
        traj.attn_slots[logical] = slot
        traj.hashes[logical] = block_hash
        return slot

    def stage_tail_states(
        self, owner: str, boundary: int, num_state_groups: int
    ) -> dict[int, int] | None:
        """Reserve slots for the tail-boundary state blocks of `owner`.

        Returns {tier_state_gid: slot} or None when capacity is unavailable.
        Replaces any previously recorded tail (a trajectory grows; its old
        tail states are superseded).
        """
        traj = self._trajectories.setdefault(owner, Trajectory())
        self.touch(owner)
        slots: dict[int, int] = {}
        for gid in range(num_state_groups):
            slot = self._alloc_slot(owner)
            if slot is None:
                for s in slots.values():
                    self._pending_write.discard(s)
                    self._free.append(s)
                return None
            slots[gid] = slot
        for s in traj.tail_state_slots.values():
            self._pending_write.discard(s)
            self._free.append(s)
        traj.tail_state_slots = slots
        traj.tail_boundary = boundary
        traj.tail_pending = True
        return slots

    def confirm_writes(self, slots: list[int]) -> None:
        for slot in slots:
            self._pending_write.discard(slot)
        for traj in self._trajectories.values():
            if traj.tail_pending and not any(
                s in self._pending_write for s in traj.tail_state_slots.values()
            ):
                traj.tail_pending = False

    # ----------------------------------------------------------------- lookup

    def lookup(
        self, hashes: list[BlockHash]
    ) -> tuple[int, list[int], dict[int, int]] | None:
        """Match `hashes` against stored trajectories.

        Returns (num_blocks, attention_slots, tail_state_slots) for the
        deepest resumable trajectory whose hash prefix matches, or None.
        """
        best: tuple[int, list[int], dict[int, int]] | None = None
        best_owner: str | None = None
        for owner, traj in list(self._trajectories.items()):
            n = traj.resumable_blocks()
            if n <= 0 or n > len(hashes):
                continue
            if best is not None and n <= best[0]:
                continue
            if any(
                s in self._pending_write for s in traj.attn_slots[:n]
            ):
                continue
            if traj.hashes[:n] != hashes[:n]:
                continue
            best = (
                n,
                [s for s in traj.attn_slots[:n] if s is not None],
                dict(traj.tail_state_slots),
            )
            best_owner = owner
        if best_owner is not None:
            self.touch(best_owner)
        return best

    def touch(self, owner: str) -> None:
        traj = self._trajectories.get(owner)
        if traj is not None:
            traj.last_touch = time.monotonic()
            self._trajectories.move_to_end(owner)

    # --------------------------------------------------------------- eviction

    def _traj_slots(self, traj: Trajectory) -> list[int]:
        return [s for s in traj.attn_slots if s is not None] + list(
            traj.tail_state_slots.values()
        )

    def _reclaim(self, protect: str) -> bool:
        for owner in list(self._trajectories.keys()):
            if owner == protect:
                continue
            traj = self._trajectories[owner]
            slots = self._traj_slots(traj)
            if any(s in self._pending_write for s in slots):
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
        if any(s in self._pending_write for s in slots):
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
                1 for t in self._trajectories.values() if t.resumable_blocks() > 0
            ),
            "pending_writes": len(self._pending_write),
        }
