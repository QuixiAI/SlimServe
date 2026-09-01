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
from collections import OrderedDict
from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class Trajectory:
    hashes: list[BlockHash] = field(default_factory=list)
    # Attention slots, position-indexed: attn_slots[i] maps each attention
    # KV-cache group id to the host slot holding that group's block i (a
    # hybrid model splits one token-block position across several groups,
    # each with its own block table - DSV4 has six). A position is complete
    # only when every attention group is present; the resumable span ends
    # at the first incomplete position.
    attn_slots: list[dict[int, int]] = field(default_factory=list)
    # Tail-boundary state: logical block index -> {tier_state_gid: slot}.
    tail_boundary: int = -1
    tail_state_slots: dict[int, int] = field(default_factory=dict)
    # Chain hash at the tail boundary (block tail_boundary - 1) recorded by
    # the request that saved the tail. Resumability requires the staged
    # attention chain to carry the SAME hash there: a tail state paired
    # with another conversation's attention blocks would resume with the
    # wrong mamba state (silent output corruption), so a mismatched
    # trajectory is simply dead rather than dangerously matchable.
    tail_hash: BlockHash = b""
    tail_pending: bool = False  # tail-state writes still in flight
    last_touch: float = 0.0

    def resumable_blocks(self, attn_gids: frozenset[int]) -> int:
        """Longest gap-free attention prefix ending at the tail boundary,
        with the boundary block's hash matching the saved tail state."""
        if self.tail_boundary <= 0 or self.tail_pending:
            return 0
        if (
            not self.tail_hash
            or self.tail_boundary > len(self.hashes)
            or self.hashes[self.tail_boundary - 1] != self.tail_hash
        ):
            return 0
        n = 0
        for slots in self.attn_slots[: self.tail_boundary]:
            if not attn_gids <= slots.keys():
                break
            n += 1
        return n if n == self.tail_boundary else 0

    def attn_prefix_len(self, attn_gids: frozenset[int]) -> int:
        """Longest gap-free staged attention prefix, tail-agnostic.

        The resumable span for attention-only models: KV blocks are
        position-independent (unlike cumulative mamba state), so any
        gap-free prefix restores validly with the remainder re-prefilled.
        """
        n = 0
        for slots in self.attn_slots:
            if not attn_gids <= slots.keys():
                break
            n += 1
        return n


class HostKVTierIndex:
    """Trajectory-centric host tier placement and lookup."""

    def __init__(self, num_slots: int, attn_gids: list[int] | None = None):
        assert num_slots > 0
        self.num_slots = num_slots
        # Attention KV-cache group ids a complete position must cover.
        self.attn_gids = frozenset(attn_gids if attn_gids is not None else [0])
        self._free: list[int] = list(range(num_slots - 1, -1, -1))
        self._trajectories: OrderedDict[str, Trajectory] = OrderedDict()
        self._pending_write: set[int] = set()
        # Slots superseded while their write was still in flight: no
        # trajectory references them any more, so free them as soon as
        # their write confirms.
        self._orphaned_pending: set[int] = set()

    # ------------------------------------------------------------------ write

    def _alloc_slot(self, protect: str) -> int | None:
        if not self._free and not self._reclaim(protect):
            return None
        slot = self._free.pop()
        self._pending_write.add(slot)
        return slot

    def stage_attention(
        self,
        owner: str,
        logical: int,
        block_hash: BlockHash,
        gid: int = 0,
        supersede: bool = False,
    ) -> int | None:
        """Reserve a slot for attention group ``gid`` of block `logical`.

        With ``supersede`` (attention-only lineages), a position already
        staged under a DIFFERENT hash means the chain diverged there (an
        adopted conversation grew past its previous generation boundary):
        the stale suffix is freed and restaged, mirroring what the GPU
        prefix cache does with a diverged tail. Without it (mamba), an
        occupied position is left untouched.
        """
        traj = self._trajectories.setdefault(owner, Trajectory())
        self.touch(owner)
        while len(traj.attn_slots) <= logical:
            traj.attn_slots.append({})
            traj.hashes.append(b"")
        if traj.hashes[logical] not in (b"", block_hash):
            if not supersede:
                return None
            # Chain diverged at `logical`: everything from here on belongs
            # to a superseded branch. Free it (a slot still pending write
            # is orphaned and freed when its write confirms).
            for i in range(logical, len(traj.attn_slots)):
                for s in traj.attn_slots[i].values():
                    if s in self._pending_write:
                        self._orphaned_pending.add(s)
                    else:
                        self._free.append(s)
                traj.attn_slots[i] = {}
                traj.hashes[i] = b""
        if gid in traj.attn_slots[logical]:
            return None
        slot = self._alloc_slot(owner)
        if slot is None:
            return None
        traj.attn_slots[logical][gid] = slot
        traj.hashes[logical] = block_hash
        return slot

    def stage_tail_states(
        self,
        owner: str,
        boundary: int,
        num_state_groups: int,
        boundary_hash: BlockHash = b"",
    ) -> dict[int, int] | None:
        """Reserve slots for the tail-boundary state blocks of `owner`.

        ``boundary_hash`` is the saver's chain hash at block
        ``boundary - 1``; resumability later requires the staged attention
        chain to carry the same hash there (see Trajectory.tail_hash).

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
        traj.tail_hash = boundary_hash
        traj.tail_pending = True
        return slots

    def confirm_writes(self, slots: list[int]) -> None:
        for slot in slots:
            self._pending_write.discard(slot)
            if slot in self._orphaned_pending:
                self._orphaned_pending.discard(slot)
                self._free.append(slot)
        for traj in self._trajectories.values():
            if traj.tail_pending and not any(
                s in self._pending_write for s in traj.tail_state_slots.values()
            ):
                traj.tail_pending = False

    # ----------------------------------------------------------------- lookup

    def lookup(
        self, hashes: list[BlockHash], allow_partial: bool = False
    ) -> tuple[str, int, list[dict[int, int]], dict[int, int]] | None:
        """Match `hashes` against stored trajectories.

        Returns (owner, num_blocks, attention_slots, tail_state_slots) for
        the deepest resumable trajectory whose hash prefix matches, or
        None. The owner lets a resuming request ADOPT the trajectory and
        extend it in place (the conversation's next turn keeps growing one
        lineage instead of duplicating it).

        With ``allow_partial`` (attention-only models: no mamba tail to
        anchor), the match is the longest common hash prefix within the
        trajectory's gap-free staged span, clamped at the first slot still
        pending write. Chat replays ALWAYS diverge at the previous turn's
        generation boundary (template-wrapped assistant text hashes
        differently from the tokens as sampled), so all-or-nothing
        matching never fires for chat traffic.
        """
        best: tuple[str, int, list[dict[int, int]], dict[int, int]] | None = None
        best_owner: str | None = None
        for owner, traj in list(self._trajectories.items()):
            if allow_partial:
                span = min(traj.attn_prefix_len(self.attn_gids), len(hashes))
                m = 0
                while (
                    m < span
                    and traj.hashes[m] == hashes[m]
                    and not any(
                        s in self._pending_write
                        for s in traj.attn_slots[m].values()
                    )
                ):
                    m += 1
                if m <= 0:
                    continue
                if best is not None and m <= best[1]:
                    continue
                best = (
                    owner,
                    m,
                    [dict(d) for d in traj.attn_slots[:m]],
                    {},
                )
                best_owner = owner
                continue
            n = traj.resumable_blocks(self.attn_gids)
            if n <= 0 or n > len(hashes):
                continue
            if best is not None and n <= best[1]:
                continue
            if any(
                s in self._pending_write
                for d in traj.attn_slots[:n]
                for s in d.values()
            ):
                continue
            if traj.hashes[:n] != hashes[:n]:
                continue
            best = (
                owner,
                n,
                [dict(d) for d in traj.attn_slots[:n]],
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
                    for i, d in enumerate(traj.attn_slots)
                    if not self.attn_gids <= d.keys()
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
                for d in traj.attn_slots
                for s in d.values()
                if s in self._pending_write
            ]
            notes.append(
                f"owner={owner[:12]} tail={traj.tail_boundary} "
                f"tail_pending={traj.tail_pending} attn_len={len(traj.attn_slots)} "
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
        return [s for d in traj.attn_slots for s in d.values()] + list(
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
                1
                for t in self._trajectories.values()
                if t.resumable_blocks(self.attn_gids) > 0
                or t.attn_prefix_len(self.attn_gids) > 0
            ),
            "pending_writes": len(self._pending_write),
        }
