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

Third tier (NVMe, optional): every host slot staged for a trajectory also
gets a disk slot, and the block is written through host -> disk once its
host copy is confirmed. A trajectory whose every resumable position is on
disk is DEMOTED rather than deleted when the host tier needs room: its host
slots are released, the disk copies stay, and the trajectory remains
resumable. A hit on a disk-only position is served by PROMOTION: fresh host
slots are allocated, the worker reads disk -> host -> GPU, and the
trajectory is host-resident again. Disk slots are reclaimed LRU per
trajectory like host slots. Positions and tails are the same logical
objects in both tiers; only their location differs.
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
    # Disk (NVMe tier) locations, mirroring the host ones position for
    # position and group for group. A copy is disk-resident when its disk
    # slot exists and is not in the index's pending-disk-write set.
    disk_attn: list[dict[int, int]] = field(default_factory=list)
    disk_tail_slots: dict[int, int] = field(default_factory=dict)
    disk_tail_boundary: int = -1

    def _attn_complete(
        self, i: int, attn_gids: frozenset[int], disk_pending: set[int]
    ) -> bool:
        """Every attention group of position i is on the host or on disk."""
        host = self.attn_slots[i] if i < len(self.attn_slots) else {}
        disk = self.disk_attn[i] if i < len(self.disk_attn) else {}
        for gid in attn_gids:
            if gid in host:
                continue
            d = disk.get(gid)
            if d is None or d in disk_pending:
                return False
        return True

    def _tail_available(self, disk_pending: set[int]) -> bool:
        if self.tail_state_slots and not self.tail_pending:
            return True
        return (
            bool(self.disk_tail_slots)
            and self.disk_tail_boundary == self.tail_boundary
            and not any(d in disk_pending for d in self.disk_tail_slots.values())
        )

    def resumable_blocks(
        self, attn_gids: frozenset[int], disk_pending: set[int] | None = None
    ) -> int:
        """Longest gap-free attention prefix ending at the tail boundary,
        with the boundary block's hash matching the saved tail state.

        A position or tail counts whether it lives on the host or on disk
        (a disk-only copy is served by promotion at lookup time)."""
        pending = disk_pending if disk_pending is not None else set()
        if self.tail_boundary <= 0 or not self._tail_available(pending):
            return 0
        if (
            not self.tail_hash
            or self.tail_boundary > len(self.hashes)
            or self.hashes[self.tail_boundary - 1] != self.tail_hash
        ):
            return 0
        n = 0
        for i in range(self.tail_boundary):
            if not self._attn_complete(i, attn_gids, pending):
                break
            n += 1
        return n if n == self.tail_boundary else 0

    def attn_prefix_len(self, attn_gids: frozenset[int]) -> int:
        """Longest gap-free HOST-staged attention prefix, tail-agnostic.

        The resumable span for attention-only models: KV blocks are
        position-independent (unlike cumulative mamba state), so any
        gap-free prefix restores validly with the remainder re-prefilled.
        Host-resident only: the partial-resume path does not promote.
        """
        n = 0
        for slots in self.attn_slots:
            if not attn_gids <= slots.keys():
                break
            n += 1
        return n

    def host_slots(self) -> list[int]:
        return [s for d in self.attn_slots for s in d.values()] + list(
            self.tail_state_slots.values()
        )

    def disk_slots(self) -> list[int]:
        return [d for m in self.disk_attn for d in m.values()] + list(
            self.disk_tail_slots.values()
        )

    def fully_on_disk(self, attn_gids: frozenset[int], disk_pending: set[int]) -> bool:
        """Every resumable position and the current tail are disk-resident,
        so the host copies can be released without losing resumability."""
        if self.tail_boundary <= 0 or self.disk_tail_boundary != self.tail_boundary:
            return False
        if not self.disk_tail_slots or any(
            d in disk_pending for d in self.disk_tail_slots.values()
        ):
            return False
        for i in range(self.tail_boundary):
            disk = self.disk_attn[i] if i < len(self.disk_attn) else {}
            for gid in attn_gids:
                d = disk.get(gid)
                if d is None or d in disk_pending:
                    return False
        return True


class HostKVTierIndex:
    """Trajectory-centric host tier placement and lookup."""

    def __init__(
        self,
        num_slots: int,
        attn_gids: list[int] | None = None,
        num_disk_slots: int = 0,
    ):
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
        # NVMe tier.
        self.num_disk_slots = max(0, int(num_disk_slots))
        self._disk_free: list[int] = list(range(self.num_disk_slots - 1, -1, -1))
        # Disk slots whose write-through has not been confirmed complete.
        self._disk_pending: set[int] = set()
        # host slot -> disk slot: write-through not yet handed to the worker.
        self._disk_of_host: dict[int, int] = {}
        # host slot -> disk slot: write-through in flight (the host slot's
        # bytes are being read by the IO engine and must stay put).
        self._host_busy: dict[int, int] = {}
        # Promotions in flight: host slots being filled from disk; confirmed
        # by the worker's restore completion for the resuming request.
        self._promotions: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ write

    def _alloc_slot(self, protect: str) -> int | None:
        if not self._free and not self._reclaim(protect):
            return None
        slot = self._free.pop()
        self._pending_write.add(slot)
        return slot

    def _alloc_disk_slot(self, protect: str) -> int | None:
        if self.num_disk_slots <= 0:
            return None
        if not self._disk_free and not self._reclaim_disk(protect):
            return None
        slot = self._disk_free.pop()
        self._disk_pending.add(slot)
        return slot

    def _free_disk_slot(self, slot: int) -> None:
        self._disk_pending.discard(slot)
        self._disk_free.append(slot)

    def _release_host_slot(self, slot: int) -> None:
        """Free a host slot and any disk copy still tied to it."""
        d = self._disk_of_host.pop(slot, None)
        if d is not None:
            self._free_disk_slot(d)
        if slot in self._pending_write:
            # Still being written: orphan it, free when the write confirms.
            self._orphaned_pending.add(slot)
        else:
            self._free.append(slot)

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
            traj.disk_attn.append({})
        if traj.hashes[logical] not in (b"", block_hash):
            if not supersede:
                return None
            # Chain diverged at `logical`: everything from here on belongs
            # to a superseded branch. Free it (a slot still pending write
            # is orphaned and freed when its write confirms).
            for i in range(logical, len(traj.attn_slots)):
                for s in traj.attn_slots[i].values():
                    self._release_host_slot(s)
                for d in traj.disk_attn[i].values():
                    self._free_disk_slot(d)
                traj.attn_slots[i] = {}
                traj.disk_attn[i] = {}
                traj.hashes[i] = b""
        if gid in traj.attn_slots[logical]:
            return None
        slot = self._alloc_slot(owner)
        if slot is None:
            return None
        traj.attn_slots[logical][gid] = slot
        traj.hashes[logical] = block_hash
        if gid in traj.disk_attn[logical]:
            # Disk-resident copy re-staged by a continuing request (a
            # demoted lineage extended without promotion): the disk copy is
            # already right; only the host copy is refilled.
            return slot
        disk = self._alloc_disk_slot(owner)
        if disk is not None:
            traj.disk_attn[logical][gid] = disk
            self._disk_of_host[slot] = disk
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
            self._disk_of_host.pop(s, None)
            self._free.append(s)
        for d in traj.disk_tail_slots.values():
            self._free_disk_slot(d)
        traj.disk_tail_slots = {}
        traj.disk_tail_boundary = -1
        traj.tail_state_slots = slots
        traj.tail_boundary = boundary
        traj.tail_hash = boundary_hash
        traj.tail_pending = True
        if self.num_disk_slots > 0:
            disk: dict[int, int] = {}
            for gid, slot in slots.items():
                d = self._alloc_disk_slot(owner)
                if d is None:
                    for dd in disk.values():
                        self._free_disk_slot(dd)
                    disk = {}
                    break
                disk[gid] = d
            for gid, d in disk.items():
                self._disk_of_host[slots[gid]] = d
            if disk:
                traj.disk_tail_slots = disk
                traj.disk_tail_boundary = boundary
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

    # ------------------------------------------------------------- disk tier

    def take_disk_writes(self, confirmed_host_slots: list[int]) -> list[tuple[int, int]]:
        """Write-through ops (host_slot, disk_slot) for host slots whose GPU ->
        host copy is confirmed. The host slot stays busy (never freed) until
        confirm_disk_writes reports the disk copy complete."""
        ops: list[tuple[int, int]] = []
        for slot in confirmed_host_slots:
            d = self._disk_of_host.pop(slot, None)
            if d is None:
                continue
            self._host_busy[slot] = d
            ops.append((slot, d))
        return ops

    def confirm_disk_writes(self, ops: list[tuple[int, int]]) -> None:
        for slot, d in ops:
            if self._host_busy.get(slot) == d:
                del self._host_busy[slot]
            self._disk_pending.discard(d)

    def needs_promotion(self, hit: tuple) -> bool:
        owner, n, attn, tail = hit
        traj = self._trajectories.get(owner)
        if traj is None:
            return False
        if any(not self.attn_gids <= d.keys() for d in attn[:n]):
            return True
        return bool(traj.tail_boundary > 0 and not tail and traj.disk_tail_slots)

    def promote(self, owner: str, n_blocks: int) -> tuple[
        list[dict[int, int]], dict[int, int], list[tuple[int, int]]
    ] | None:
        """Give every disk-only copy of `owner`'s first `n_blocks` positions
        (and its tail) a host slot to be filled from disk.

        Returns (attn_host_slots, tail_host_slots, disk_reads) where
        disk_reads are (disk_slot, host_slot) pairs the worker must complete
        before the restore copies; the new host slots stay pending until
        confirm_promotion(owner). None on host-capacity failure (rolled
        back: the trajectory stays disk-resident, the request misses).
        """
        traj = self._trajectories.get(owner)
        if traj is None:
            return None
        reads: list[tuple[int, int]] = []
        new_slots: list[int] = []
        attn: list[dict[int, int]] = []
        for i in range(n_blocks):
            host = traj.attn_slots[i]
            for gid in self.attn_gids:
                if gid in host:
                    continue
                d = traj.disk_attn[i].get(gid)
                assert d is not None and d not in self._disk_pending
                s = self._alloc_slot(owner)
                if s is None:
                    self._rollback_promotion(traj, new_slots)
                    return None
                host[gid] = s
                new_slots.append(s)
                reads.append((d, s))
            attn.append(dict(host))
        tail: dict[int, int] = dict(traj.tail_state_slots)
        if not tail or traj.tail_pending:
            tail = {}
            for gid, d in traj.disk_tail_slots.items():
                s = self._alloc_slot(owner)
                if s is None:
                    self._rollback_promotion(traj, new_slots + list(tail.values()))
                    return None
                tail[gid] = s
                new_slots.append(s)
                reads.append((d, s))
            traj.tail_state_slots = tail
            traj.tail_pending = bool(reads)
        if new_slots:
            self._promotions.setdefault(owner, []).extend(new_slots)
        return attn, tail, reads

    def _rollback_promotion(self, traj: Trajectory, slots: list[int]) -> None:
        drop = set(slots)
        for d in traj.attn_slots:
            for gid in [g for g, s in d.items() if s in drop]:
                del d[gid]
        for gid in [g for g, s in traj.tail_state_slots.items() if s in drop]:
            del traj.tail_state_slots[gid]
        for s in slots:
            self._pending_write.discard(s)
            self._free.append(s)

    def confirm_promotion(self, owner: str) -> None:
        """The worker completed the disk -> host reads for `owner`'s resume."""
        slots = self._promotions.pop(owner, None)
        if slots:
            self.confirm_writes(slots)

    def abort_promotion(self, owner: str) -> None:
        traj = self._trajectories.get(owner)
        slots = self._promotions.pop(owner, None)
        if traj is not None and slots:
            self._rollback_promotion(traj, slots)

    # ----------------------------------------------------------------- lookup

    def lookup(
        self, hashes: list[BlockHash], allow_partial: bool = False
    ) -> tuple[str, int, list[dict[int, int]], dict[int, int]] | None:
        """Match `hashes` against stored trajectories.

        Returns (owner, num_blocks, attention_slots, tail_state_slots) for
        the deepest resumable trajectory whose hash prefix matches, or
        None. The owner lets a resuming request ADOPT the trajectory and
        extend it in place (the conversation's next turn keeps growing one
        lineage instead of duplicating it). A position dict missing an
        attention group (or an empty tail on a hybrid trajectory) is
        disk-only: call promote().

        With ``allow_partial`` (attention-only models: no mamba tail to
        anchor), the match is the longest common hash prefix within the
        trajectory's gap-free HOST-staged span, clamped at the first slot
        still pending write. Chat replays ALWAYS diverge at the previous
        turn's generation boundary (template-wrapped assistant text hashes
        differently from the tokens as sampled), so all-or-nothing
        matching never fires for chat traffic.
        """
        best: tuple[str, int, list[dict[int, int]], dict[int, int]] | None = None
        best_owner: str | None = None
        for owner, traj in list(self._trajectories.items()):
            if owner in self._promotions:
                continue
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
            n = traj.resumable_blocks(self.attn_gids, self._disk_pending)
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
            tail = {} if traj.tail_pending else dict(traj.tail_state_slots)
            best = (
                owner,
                n,
                [dict(d) for d in traj.attn_slots[:n]],
                tail,
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
                    for i in range(len(traj.attn_slots))
                    if not traj._attn_complete(i, self.attn_gids, self._disk_pending)
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
                f"pending_attn={len(pend)} req_hashes={len(hashes)} "
                f"on_disk={traj.fully_on_disk(self.attn_gids, self._disk_pending)}"
            )
        return "; ".join(notes) or "no trajectory shares hashes[0]"

    def touch(self, owner: str) -> None:
        traj = self._trajectories.get(owner)
        if traj is not None:
            traj.last_touch = time.monotonic()
            self._trajectories.move_to_end(owner)

    # --------------------------------------------------------------- eviction

    def _traj_slots(self, traj: Trajectory) -> list[int]:
        return traj.host_slots()

    def _busy(self, traj: Trajectory) -> bool:
        return any(
            s in self._pending_write or s in self._host_busy for s in traj.host_slots()
        )

    def _delete(self, owner: str, traj: Trajectory) -> None:
        for s in traj.host_slots():
            self._pending_write.discard(s)
            self._disk_of_host.pop(s, None)
            self._free.append(s)
        for d in traj.disk_slots():
            self._free_disk_slot(d)
        del self._trajectories[owner]

    def _reclaim(self, protect: str) -> bool:
        """Free the host slots of the coldest trajectory: demote it when
        every resumable copy is on disk, delete it otherwise."""
        for owner in list(self._trajectories.keys()):
            if owner == protect or owner in self._promotions:
                continue
            traj = self._trajectories[owner]
            if self._busy(traj):
                continue
            host = traj.host_slots()
            if not host:
                continue  # already disk-only; nothing to free here
            if traj.fully_on_disk(self.attn_gids, self._disk_pending):
                for s in host:
                    self._disk_of_host.pop(s, None)
                    self._free.append(s)
                traj.attn_slots = [{} for _ in traj.attn_slots]
                traj.tail_state_slots = {}
                traj.tail_pending = False
            else:
                self._delete(owner, traj)
            return True
        return False

    def _reclaim_disk(self, protect: str) -> bool:
        """Free the disk slots of the coldest trajectory holding any; a
        disk-only trajectory dies with them."""
        busy_disk = set(self._host_busy.values())
        for owner in list(self._trajectories.keys()):
            if owner == protect or owner in self._promotions:
                continue
            traj = self._trajectories[owner]
            disk = traj.disk_slots()
            if not disk or any(d in self._disk_pending or d in busy_disk for d in disk):
                continue
            if not traj.host_slots():
                self._delete(owner, traj)
                return True
            for d in disk:
                self._free_disk_slot(d)
            dropped = set(disk)
            for s in [s for s, d in self._disk_of_host.items() if d in dropped]:
                del self._disk_of_host[s]
            traj.disk_attn = [{} for _ in traj.disk_attn]
            traj.disk_tail_slots = {}
            traj.disk_tail_boundary = -1
            return True
        return False

    def drop_owner(self, owner: str) -> None:
        traj = self._trajectories.get(owner)
        if traj is None:
            return
        if self._busy(traj) or owner in self._promotions:
            return
        if any(d in self._disk_pending for d in traj.disk_slots()):
            return
        self._delete(owner, traj)

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict[str, int]:
        return {
            "slots": self.num_slots,
            "used": self.num_slots - len(self._free),
            "trajectories": len(self._trajectories),
            "resumable": sum(
                1
                for t in self._trajectories.values()
                if t.resumable_blocks(self.attn_gids, self._disk_pending) > 0
                or t.attn_prefix_len(self.attn_gids) > 0
            ),
            "pending_writes": len(self._pending_write),
            "disk_slots": self.num_disk_slots,
            "disk_used": self.num_disk_slots - len(self._disk_free),
            "disk_pending": len(self._disk_pending),
            "disk_only": sum(
                1
                for t in self._trajectories.values()
                if not t.host_slots() and t.disk_slots()
            ),
        }
