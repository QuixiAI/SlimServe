# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side index for the host KV tier (trajectory-centric).

The host tier mirrors a trajectory's immutable full-attention blocks at
fill time and its mamba/state tail-boundary blocks when the request
finishes. A trajectory is resumable only at its recorded tail boundary:
mamba (align-mode) state exists only there, mirroring the engine's own
resident-state semantics (earlier boundary positions are nulls).

Lookups match a new request's hash chain against a stored trajectory: a hit
restores attention blocks [0, tail) plus the tail state blocks, and the
request resumes computing from the tail boundary.

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
trajectories, never individual slots.
"""

from __future__ import annotations

import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import BlockHash

_EMPTY_HASH = BlockHash(b"")


@dataclass
class Trajectory:
    hashes: list[BlockHash] = field(default_factory=list)
    # Attention slots, position-indexed PER GROUP: attn_slots[g][p] holds
    # attention-group ordinal g's block at position p (positions are in
    # that group's own page granularity). Group 0 is the primary
    # full-attention group - its positions align 1:1 with the hash chain
    # and gate resumability. Non-primary groups (sliding-window MLA pages
    # etc.) may legally have missing positions: window-skipped pages are
    # never staged, and the connector pairs them against null target
    # blocks on resume.
    attn_slots: list[list[int | None]] = field(default_factory=list)
    # Tail-boundary state: logical block index -> {tier_state_gid: slot}.
    tail_boundary: int = -1
    tail_state_slots: dict[int, int] = field(default_factory=dict)
    # Chain hash at the tail boundary (block tail_boundary - 1) recorded by
    # the request that saved the tail. Resumability requires the staged
    # attention chain to carry the SAME hash there: a tail state paired
    # with another conversation's attention blocks would resume with the
    # wrong mamba state (silent output corruption), so a mismatched
    # trajectory is simply dead rather than dangerously matchable.
    tail_hash: BlockHash = BlockHash(b"")
    tail_pending: bool = False  # tail-state writes still in flight
    last_touch: float = 0.0

    def primary_slots(self) -> list[int | None]:
        return self.attn_slots[0] if self.attn_slots else []

    def resumable_blocks(self) -> int:
        """Longest gap-free PRIMARY-group prefix ending at the tail
        boundary, with the boundary block's hash matching the saved tail
        state. Non-primary groups do not gate the span: their missing
        positions are window skips by construction."""
        if self.tail_boundary <= 0 or self.tail_pending:
            return 0
        if (
            not self.tail_hash
            or self.tail_boundary > len(self.hashes)
            or self.hashes[self.tail_boundary - 1] != self.tail_hash
        ):
            return 0
        n = 0
        for slot in self.primary_slots()[: self.tail_boundary]:
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
        self._readers: Counter[str] = Counter()

    # ------------------------------------------------------------------ write

    def _alloc_slot(self, protect: str) -> int | None:
        if not self._free and not self._reclaim(protect):
            return None
        slot = self._free.pop()
        self._pending_write.add(slot)
        return slot

    def stage_attention(
        self, owner: str, gid: int, pos: int, block_hash: BlockHash = _EMPTY_HASH
    ) -> int | None:
        """Reserve a slot for attention-group ordinal ``gid``'s page at
        ``pos`` (the group's own granularity). The hash chain is recorded
        only from the primary group (gid 0), whose positions align with
        the hash blocks."""
        traj = self._trajectories.setdefault(owner, Trajectory())
        self.touch(owner)
        while len(traj.attn_slots) <= gid:
            traj.attn_slots.append([])
        row = traj.attn_slots[gid]
        while len(row) <= pos:
            row.append(None)
        if gid == 0:
            while len(traj.hashes) <= pos:
                traj.hashes.append(BlockHash(b""))
        if row[pos] is not None:
            return None
        slot = self._alloc_slot(owner)
        if slot is None:
            return None
        row[pos] = slot
        if gid == 0:
            traj.hashes[pos] = block_hash
        return slot

    def stage_tail_states(
        self,
        owner: str,
        boundary: int,
        num_state_groups: int,
        boundary_hash: BlockHash = _EMPTY_HASH,
    ) -> dict[int, int] | None:
        """Reserve slots for the tail-boundary state blocks of `owner`.

        ``boundary_hash`` is the saver's chain hash at block
        ``boundary - 1``; resumability later requires the staged attention
        chain to carry the same hash there (see Trajectory.tail_hash).

        Returns {tier_state_gid: slot} or None when capacity is unavailable.
        Replaces any previously recorded tail (a trajectory grows; its old
        tail states are superseded).
        """
        if self._readers[owner]:
            # A concurrent continuation may still be reading the old tail.
            # Keep that snapshot until its restore completes.
            return None
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
        traj.tail_hash = boundary_hash
        traj.tail_pending = True
        return slots

    def free_attention(self, owner: str, gid: int, pos: int) -> bool:
        """Release a non-primary group's staged page whose position slid out
        of the group's window (the engine's allocation went null there).
        Without this, a sliding-window group permanently holds one slot for
        every page that was EVER in-window - measured 11k slots per 23k-token
        request on DSV4/Metal, 120x the primary cost - and the LRU then
        reclaims whole live trajectories. Skipped while the write is still
        in flight (rare; the slot then rides until the trajectory dies)."""
        if self._readers[owner]:
            return False
        traj = self._trajectories.get(owner)
        if traj is None or gid <= 0 or gid >= len(traj.attn_slots):
            return False
        row = traj.attn_slots[gid]
        slot = row[pos] if pos < len(row) else None
        if slot is None or slot in self._pending_write:
            return False
        row[pos] = None
        self._free.append(slot)
        return True

    def stage_attention_tail(self, owner: str) -> int:
        """Record the resume boundary for a STATELESS (attention-only)
        trajectory at its current gap-free staged span.

        Models with no recurrent state need no boundary snapshot - any
        gap-free attention prefix is self-sufficient - but ``lookup``
        still gates on ``tail_boundary``/``tail_hash`` (they are what
        binds a resume to one conversation's chain). Without this call an
        attention-only trajectory is write-only: staged, never
        resumable. No slots are consumed. Returns the boundary (0 = not
        stageable yet).
        """
        traj = self._trajectories.get(owner)
        if traj is None:
            return 0
        if traj.tail_state_slots and self._readers[owner]:
            # A lifecycle conversion cannot recycle an active reader's
            # state snapshot, just like ordinary tail replacement.
            return 0
        self.touch(owner)
        n = 0
        for slot in traj.primary_slots():
            if slot is None:
                break
            n += 1
        if n <= 0:
            return 0
        for s in traj.tail_state_slots.values():
            self._pending_write.discard(s)
            self._free.append(s)
        traj.tail_state_slots = {}
        traj.tail_boundary = n
        traj.tail_hash = traj.hashes[n - 1]
        # No state writes exist to be in flight; in-flight ATTENTION
        # writes are already excluded per-slot by lookup's pending check.
        traj.tail_pending = False
        return n

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
    ) -> tuple[str, int, list[list[int | None]], dict[int, int]] | None:
        """Match `hashes` against stored trajectories.

        Returns (owner, num_blocks, attn_group_slots, tail_state_slots)
        for the deepest resumable trajectory whose hash prefix matches, or
        None. ``attn_group_slots[g]`` is group ordinal g's positional slot
        list (None marks positions that were never staged - window skips
        on non-primary groups); the primary list is gap-free for the first
        ``num_blocks`` positions. The owner lets a resuming request ADOPT
        the trajectory and extend it in place (the conversation's next
        turn keeps growing one lineage instead of duplicating it).
        """
        best: tuple[str, int, list[list[int | None]], dict[int, int]] | None = None
        best_owner: str | None = None
        for owner, traj in list(self._trajectories.items()):
            n = traj.resumable_blocks()
            if n <= 0:
                continue
            if not traj.tail_state_slots:
                # STATELESS trajectory: any gap-free prefix is valid, so
                # match as deep as the hashes agree. This is the chat
                # resume shape - a saved tail routinely crosses into the
                # request's own generated thinking tokens, which no
                # follow-up prompt resends, so tail-exact matching would
                # deadletter most conversations (observed live: tail=4
                # matched 3 prompt blocks, req_hashes=3).
                limit = min(n, len(hashes))
                n = 0
                while n < limit and traj.hashes[n] == hashes[n]:
                    n += 1
                if n <= 0:
                    continue
            elif n > len(hashes) or traj.hashes[:n] != hashes[:n]:
                # STATEFUL trajectory: mamba state exists only at the tail
                # boundary, so resume is tail-exact or nothing.
                continue
            if best is not None and n <= best[1]:
                continue
            if any(
                s in self._pending_write
                for row in traj.attn_slots
                for s in row
                if s is not None
            ):
                continue
            best = (
                owner,
                n,
                [list(row) for row in traj.attn_slots],
                dict(traj.tail_state_slots),
            )
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
            gap = next(
                (
                    i
                    for i, s in enumerate(traj.primary_slots()[: traj.tail_boundary])
                    if s is None
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
                for s in traj.primary_slots()[: traj.tail_boundary]
                if s in self._pending_write
            ]
            notes.append(
                f"owner={owner[:12]} tail={traj.tail_boundary} "
                f"tail_pending={traj.tail_pending} "
                f"attn_len={len(traj.primary_slots())} "
                f"first_gap={gap} hash_mismatch_at={mism} "
                f"pending_attn={len(pend)} req_hashes={len(hashes)}"
            )
        return "; ".join(notes) or "no trajectory shares hashes[0]"

    def touch(self, owner: str) -> None:
        traj = self._trajectories.get(owner)
        if traj is not None:
            traj.last_touch = time.monotonic()
            self._trajectories.move_to_end(owner)

    def pin_read(self, owner: str) -> None:
        """Hold immutable source slots from lookup through restore completion."""
        assert owner in self._trajectories
        self._readers[owner] += 1

    def can_extend(self, owner: str, boundary: int) -> bool:
        """Appending is safe only when no saved suffix diverges after the hit."""
        traj = self._trajectories.get(owner)
        return traj is not None and boundary == len(traj.hashes)

    def is_read_pinned(self, owner: str) -> bool:
        return self._readers[owner] > 0

    def unpin_read(self, owner: str) -> None:
        assert self._readers[owner] > 0
        self._readers[owner] -= 1
        if not self._readers[owner]:
            del self._readers[owner]

    # --------------------------------------------------------------- eviction

    def _traj_slots(self, traj: Trajectory) -> list[int]:
        return [s for row in traj.attn_slots for s in row if s is not None] + list(
            traj.tail_state_slots.values()
        )

    def _reclaim(self, protect: str) -> bool:
        for owner in list(self._trajectories.keys()):
            if owner == protect or self._readers[owner]:
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
        if self._readers[owner]:
            # Invalidate future matches without recycling an active source.
            traj.tail_boundary = -1
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
