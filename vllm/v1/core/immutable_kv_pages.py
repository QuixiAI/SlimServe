# SPDX-License-Identifier: Apache-2.0
"""Physical-page ownership for a future content-addressed KV tier.

Not wired into HostKVTierIndex yet. One pool belongs to one engine and one
fixed group/layout namespace. Full attention pages may share a key; Mamba
tails must use ``acquire_private`` and retain trajectory-local hash checks.

This module owns slots and lifetimes, not bytes, trajectories or GPU blocks.
The connector must pin GPU source/destination blocks separately and submit
each ticket exactly once when handing its work to the workers. Completion
means ALL relevant workers have finished. Submitted tickets cannot be
cancelled early, even after every logical owner disappears.
Methods run on the scheduler thread; IO threads enqueue completion messages
instead of mutating this pool directly.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["host", "disk"]
Operation = Literal["offload", "writeback", "promote", "restore"]


@dataclass(frozen=True, slots=True)
class AttentionPageKey:
    group_id: int
    block_hash: bytes

    def __post_init__(self) -> None:
        if type(self.group_id) is not int or self.group_id < 0:
            raise ValueError("group_id must be a nonnegative integer")
        if not isinstance(self.block_hash, bytes) or not self.block_hash:
            raise ValueError("a nonempty immutable chain hash is required")


@dataclass(frozen=True, slots=True)
class PageLease:
    """One attachment, not a raw slot. Only its creating pool accepts it."""

    lease_id: int
    page_id: int
    owner: str


@dataclass(frozen=True, slots=True)
class PageTicket:
    """Reserved transfer/read. Locations stay stable until finish/cancel."""

    ticket_id: int
    operation: Operation
    source: tuple[Tier, int] | None
    destination: tuple[Tier, int] | None


@dataclass(frozen=True, slots=True)
class PageView:
    page_id: int
    references: int
    host_slot: int | None
    disk_slot: int | None
    host_ready: bool
    disk_ready: bool


@dataclass(slots=True)
class _Copy:
    tier: Tier
    slot: int
    ready: bool = False
    pins: int = 0


@dataclass(slots=True)
class _Page:
    page_id: int
    key: AttentionPageKey | None
    references: int = 0
    host: _Copy | None = None
    disk: _Copy | None = None


@dataclass(slots=True)
class _Pending:
    ticket: PageTicket
    page: _Page
    source: _Copy | None
    destination: _Copy | None
    submitted: bool = False


class ImmutableKVPagePool:
    """Reference-counted host/disk pages with pinned two-phase transfers.

    A reservation returning None means no new operation: the copy may
    already exist (possibly pending), or capacity/source readiness may be
    unavailable. ``view`` distinguishes these cases. Callers must not treat
    a pending copy as readable or acknowledge an unscheduled operation.

    Capacity reclaim drops only an unpinned, ready redundant copy. If the
    other tier has no ready copy, the trajectory index must release cold
    owners instead. Releasing a lease is idempotent and cannot release a
    different owner's reference. Active tickets survive their last lease.
    """

    def __init__(self, host_slots: int, disk_slots: int = 0):
        if type(host_slots) is not int or host_slots < 1:
            raise ValueError("host_slots must be a positive integer")
        if type(disk_slots) is not int or disk_slots < 0:
            raise ValueError("disk_slots must be a nonnegative integer")
        self._capacity = {"host": host_slots, "disk": disk_slots}
        self._free = {
            tier: list(range(count - 1, -1, -1))
            for tier, count in self._capacity.items()
        }
        self._copies: dict[str, dict[int, _Copy]] = {"host": {}, "disk": {}}
        self._pages: OrderedDict[int, _Page] = OrderedDict()
        self._canonical: dict[AttentionPageKey, _Page] = {}
        self._leases: dict[int, tuple[PageLease, _Page]] = {}
        self._pending: dict[int, _Pending] = {}
        self._next_page = self._next_lease = self._next_ticket = 0

    def acquire_attention(self, key: AttentionPageKey, owner: str) -> PageLease:
        if not isinstance(key, AttentionPageKey):
            raise TypeError("attention pages require an AttentionPageKey")
        return self._acquire(key, owner)

    def acquire_private(self, owner: str) -> PageLease:
        """Always create a distinct page, even for the same owner/hash tail."""
        return self._acquire(None, owner)

    def _acquire(self, key: AttentionPageKey | None, owner: str) -> PageLease:
        if not isinstance(owner, str) or not owner:
            raise ValueError("a nonempty owner is required")
        page = self._canonical.get(key) if key is not None else None
        if page is None:
            page = _Page(self._next_page, key)
            self._next_page += 1
            self._pages[page.page_id] = page
            if key is not None:
                self._canonical[key] = page
        lease = PageLease(self._next_lease, page.page_id, owner)
        self._next_lease += 1
        self._leases[lease.lease_id] = lease, page
        page.references += 1
        self._pages.move_to_end(page.page_id)
        return lease

    def _page_for(self, lease: PageLease) -> _Page:
        entry = self._leases.get(lease.lease_id)
        if entry is None or entry[0] is not lease:
            raise ValueError("released, foreign or forged page lease")
        return entry[1]

    def release(self, lease: PageLease) -> bool:
        entry = self._leases.get(lease.lease_id)
        if entry is None or entry[0] is not lease:
            return False
        _, page = self._leases.pop(lease.lease_id)
        page.references -= 1
        self._collect(page)
        return True

    def view(self, lease: PageLease) -> PageView:
        page = self._page_for(lease)
        return PageView(
            page.page_id,
            page.references,
            page.host.slot if page.host else None,
            page.disk.slot if page.disk else None,
            bool(page.host and page.host.ready),
            bool(page.disk and page.disk.ready),
        )

    def begin_offload(self, lease: PageLease) -> PageTicket | None:
        page = self._page_for(lease)
        if page.host is not None:
            return None
        destination = self._allocate(page, "host")
        if destination is None:
            return None
        return self._reserve(page, "offload", None, destination)

    def begin_writeback(self, lease: PageLease) -> PageTicket | None:
        page = self._page_for(lease)
        if page.disk is not None or page.host is None or not page.host.ready:
            return None
        destination = self._allocate(page, "disk")
        if destination is None:
            return None
        return self._reserve(page, "writeback", page.host, destination)

    def begin_promotion(self, lease: PageLease) -> PageTicket | None:
        page = self._page_for(lease)
        if page.host is not None or page.disk is None or not page.disk.ready:
            return None
        destination = self._allocate(page, "host")
        if destination is None:
            return None
        return self._reserve(page, "promote", page.disk, destination)

    def begin_restore(self, lease: PageLease) -> PageTicket | None:
        """Pin a ready host page before queuing its host -> GPU restore."""
        page = self._page_for(lease)
        if page.host is None or not page.host.ready:
            return None
        return self._reserve(page, "restore", page.host, None)

    def _reserve(
        self,
        page: _Page,
        operation: Operation,
        source: _Copy | None,
        destination: _Copy | None,
    ) -> PageTicket:
        ticket = PageTicket(
            self._next_ticket,
            operation,
            (source.tier, source.slot) if source else None,
            (destination.tier, destination.slot) if destination else None,
        )
        self._next_ticket += 1
        for copy in (source, destination):
            if copy is not None:
                copy.pins += 1
        self._pending[ticket.ticket_id] = _Pending(ticket, page, source, destination)
        self._pages.move_to_end(page.page_id)
        return ticket

    def _operation_for(self, ticket: PageTicket) -> _Pending:
        pending = self._pending.get(ticket.ticket_id)
        if pending is None or pending.ticket is not ticket:
            raise ValueError("finished, foreign or forged page ticket")
        return pending

    def submit(self, ticket: PageTicket) -> None:
        pending = self._operation_for(ticket)
        if pending.submitted:
            raise ValueError("page ticket was already submitted")
        pending.submitted = True

    def _validate_many(
        self, tickets: Sequence[PageTicket], *, submitted: bool
    ) -> list[_Pending]:
        if not tickets or len({t.ticket_id for t in tickets}) != len(tickets):
            raise ValueError("a nonempty batch of distinct page tickets is required")
        pending = [self._operation_for(ticket) for ticket in tickets]
        if any(p.submitted != submitted for p in pending):
            raise ValueError("page ticket batch has an invalid submission state")
        return pending

    def submit_all(self, tickets: Sequence[PageTicket]) -> None:
        """Validate the entire batch before changing any ticket's state."""
        pending = self._validate_many(tickets, submitted=False)
        for operation in pending:
            operation.submitted = True

    def complete_all(self, tickets: Sequence[PageTicket], *, success: bool) -> None:
        pending = self._validate_many(tickets, submitted=True)
        for operation in pending:
            self._finish(operation, success=success)

    def cancel_all(self, tickets: Sequence[PageTicket]) -> None:
        pending = self._validate_many(tickets, submitted=False)
        for operation in pending:
            self._finish(operation, success=False)

    def cancel(self, ticket: PageTicket) -> None:
        pending = self._operation_for(ticket)
        if pending.submitted:
            raise ValueError("submitted work must finish before releasing its slots")
        self._finish(pending, success=False)

    def complete(self, ticket: PageTicket, *, success: bool = True) -> None:
        pending = self._operation_for(ticket)
        if not pending.submitted:
            raise ValueError("cannot complete an unsubmitted page ticket")
        self._finish(pending, success=success)

    def _finish(self, pending: _Pending, *, success: bool) -> None:
        del self._pending[pending.ticket.ticket_id]
        for copy in (pending.source, pending.destination):
            if copy is not None:
                copy.pins -= 1
        destination = pending.destination
        if destination is not None:
            if success:
                destination.ready = True
            else:
                self._drop_copy(pending.page, destination.tier)
        self._collect(pending.page)

    def _allocate(self, page: _Page, tier: Tier) -> _Copy | None:
        free = self._free[tier]
        if not free:
            for candidate in self._pages.values():
                if self._evict_copy(candidate, tier):
                    break
        if not free:
            return None
        copy = _Copy(tier, free.pop())
        if copy.slot in self._copies[tier]:
            raise RuntimeError("allocated a slot which already has an owner")
        self._copies[tier][copy.slot] = copy
        setattr(page, tier, copy)
        return copy

    def _evict_copy(self, page: _Page, tier: Tier) -> bool:
        copy = getattr(page, tier)
        other = page.disk if tier == "host" else page.host
        if (
            copy is None
            or not copy.ready
            or copy.pins
            or other is None
            or not other.ready
        ):
            return False
        self._drop_copy(page, tier)
        return True

    def evict_host(self, lease: PageLease) -> bool:
        """Drop one redundant host copy, preserving every logical owner."""
        return self._evict_copy(self._page_for(lease), "host")

    def evict_disk(self, lease: PageLease) -> bool:
        return self._evict_copy(self._page_for(lease), "disk")

    def _drop_copy(self, page: _Page, tier: Tier) -> None:
        copy = getattr(page, tier)
        if copy is None or copy.pins:
            raise RuntimeError("cannot free a missing or pinned page copy")
        if self._copies[tier].get(copy.slot) is not copy:
            raise RuntimeError("page slot ownership is inconsistent")
        del self._copies[tier][copy.slot]
        self._free[tier].append(copy.slot)
        setattr(page, tier, None)

    def _collect(self, page: _Page) -> None:
        if page.references or any(c and c.pins for c in (page.host, page.disk)):
            return
        for tier in ("host", "disk"):
            if getattr(page, tier) is not None:
                self._drop_copy(page, tier)
        if page.key is not None:
            del self._canonical[page.key]
        del self._pages[page.page_id]

    def stats(self) -> dict[str, int]:
        return {
            "pages": len(self._pages),
            "attention_pages": len(self._canonical),
            "leases": len(self._leases),
            "host_used": len(self._copies["host"]),
            "disk_used": len(self._copies["disk"]),
            "pending": len(self._pending),
        }

    def check_invariants(self) -> None:
        """Expensive CPU diagnostic; never call on the serving hot path."""
        refs = Counter(page.page_id for _, page in self._leases.values())
        for lease, page in self._leases.values():
            assert self._pages[lease.page_id] is page
            assert lease.page_id == page.page_id
        pins = Counter()
        for pending in self._pending.values():
            assert self._pages[pending.page.page_id] is pending.page
            for copy in (pending.source, pending.destination):
                if copy is not None:
                    assert getattr(pending.page, copy.tier) is copy
                    pins[(copy.tier, copy.slot)] += 1
            if pending.source:
                assert pending.source.ready
            if pending.destination:
                assert not pending.destination.ready
        for tier, capacity in self._capacity.items():
            free = self._free[tier]
            allocated = self._copies[tier]
            assert len(free) == len(set(free))
            assert not set(free) & allocated.keys()
            assert set(free) | allocated.keys() == set(range(capacity))
            physical_copies = [
                getattr(page, tier)
                for page in self._pages.values()
                if getattr(page, tier) is not None
            ]
            assert len(physical_copies) == len(allocated)
            page_copies = {copy.slot: copy for copy in physical_copies}
            assert len(page_copies) == len(allocated)
            assert all(page_copies[slot] is copy for slot, copy in allocated.items())
            assert all(
                copy.pins == pins[(tier, slot)] for slot, copy in allocated.items()
            )
        for page in self._pages.values():
            assert page.references == refs[page.page_id]
            assert page.references or any(c and c.pins for c in (page.host, page.disk))
            if page.key is not None:
                assert self._canonical[page.key] is page
        assert len(self._canonical) == sum(
            p.key is not None for p in self._pages.values()
        )


@dataclass(slots=True)
class _RankBatch:
    tickets: tuple[PageTicket, ...]
    ranks: dict[int, bool] = field(default_factory=dict)


class PageCompletionBarrier:
    """Scheduler-side all-rank completion for physical-page ticket batches.

    Call submit when dispatching metadata, not when merely reserving slots.
    This barrier then owns completion of those tickets; callers must not
    complete them separately. One monotonically increasing batch namespace
    covers every operation type. Rank ACKs are final outcomes, not progress
    or requests to cancel. A failed rank does not unpin other ranks' IO.
    Not connected to the existing HostTierStats wire protocol yet.
    """

    def __init__(self, pool: ImmutableKVPagePool, num_ranks: int):
        if type(num_ranks) is not int or num_ranks < 1:
            raise ValueError("num_ranks must be a positive integer")
        self._pool = pool
        self._num_ranks = num_ranks
        self._next_batch = 0
        self._batches: dict[int, _RankBatch] = {}

    def submit(self, tickets: Sequence[PageTicket]) -> int:
        tickets = tuple(tickets)
        self._pool.submit_all(tickets)
        batch_id = self._next_batch
        self._next_batch += 1
        self._batches[batch_id] = _RankBatch(tickets)
        return batch_id

    def acknowledge(self, batch_id: int, rank: int, *, success: bool = True) -> bool:
        """Return True only when this ACK finishes a whole batch.

        Duplicate final ACKs are ignored. Contradictory outcomes while a
        batch is pending indicate a protocol error. Retired IDs need no
        permanent tombstones because IDs are never reused within this engine.
        """
        if type(rank) is not int or not 0 <= rank < self._num_ranks:
            raise ValueError("rank is outside this completion group")
        if type(success) is not bool:
            raise ValueError("success must be a final boolean outcome")
        if type(batch_id) is not int or not 0 <= batch_id < self._next_batch:
            raise ValueError("unknown completion batch")
        batch = self._batches.get(batch_id)
        if batch is None:
            return False
        if rank in batch.ranks:
            if batch.ranks[rank] != success:
                raise ValueError("conflicting completion outcomes from one rank")
            return False
        batch.ranks[rank] = success
        if len(batch.ranks) != self._num_ranks:
            return False
        self._pool.complete_all(batch.tickets, success=all(batch.ranks.values()))
        del self._batches[batch_id]
        return True

    def pending(self) -> int:
        return len(self._batches)
