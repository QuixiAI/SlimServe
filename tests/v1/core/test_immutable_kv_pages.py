# SPDX-License-Identifier: Apache-2.0
"""CPU byte/lifetime checks; this is not connector or GPU qualification."""

import random
from dataclasses import replace

import pytest

from vllm.v1.core.immutable_kv_pages import AttentionPageKey, ImmutableKVPagePool


def key(value=0, group=0):
    return AttentionPageKey(group, str(value).encode())


class Worker:
    """Delayed mock IO reads actual current slot contents, exposing aliasing."""

    def __init__(self, pool):
        self.pool = pool
        self.bytes = {}

    def finish(self, ticket, value, *, success=True):
        assert ticket is not None
        if success:
            if ticket.source is not None:
                assert self.bytes[ticket.source] == value, "source slot was overwritten"
            if ticket.destination is not None:
                self.bytes[ticket.destination] = value
        self.pool.complete(ticket, success=success)

    def run(self, ticket, value):
        assert ticket is not None
        self.pool.submit(ticket)
        self.finish(ticket, value)

    def store(self, lease, value, *, disk=True):
        self.run(self.pool.begin_offload(lease), value)
        if disk:
            self.run(self.pool.begin_writeback(lease), value)


def test_pending_attention_copy_is_shared_but_not_readable():
    pool = ImmutableKVPagePool(1, 1)
    worker = Worker(pool)
    a = pool.acquire_attention(key(), "a")
    ticket = pool.begin_offload(a)
    b = pool.acquire_attention(key(), "b")
    assert pool.view(a).page_id == pool.view(b).page_id
    assert pool.view(b).references == 2
    assert pool.begin_offload(b) is None
    assert pool.begin_restore(b) is None
    assert not pool.view(b).host_ready
    worker.run(ticket, b"attention")
    worker.run(pool.begin_restore(b), b"attention")
    worker.run(pool.begin_writeback(b), b"attention")
    assert pool.begin_writeback(a) is None
    assert pool.release(a)
    assert not pool.release(a)
    assert pool.view(b).references == 1
    worker.run(pool.begin_restore(b), b"attention")
    assert pool.release(b)
    assert pool.stats() == dict(
        pages=0, attention_pages=0, leases=0, host_used=0, disk_used=0, pending=0
    )
    pool.check_invariants()


def test_private_tails_groups_and_diverged_hashes_do_not_alias():
    pool = ImmutableKVPagePool(6)
    leases = [
        pool.acquire_private("same-owner"),
        pool.acquire_private("same-owner"),
        pool.acquire_attention(key(0, 0), "a"),
        pool.acquire_attention(key(0, 1), "a"),
        pool.acquire_attention(key(1, 0), "a"),
    ]
    worker = Worker(pool)
    for i, lease in enumerate(leases):
        worker.store(lease, str(i).encode(), disk=False)
    assert len({pool.view(x).page_id for x in leases}) == len(leases)
    assert len({pool.view(x).host_slot for x in leases}) == len(leases)
    pool.check_invariants()


@pytest.mark.parametrize("revive", [False, True])
def test_last_lease_release_keeps_submitted_offload_alive(revive):
    pool = ImmutableKVPagePool(1)
    worker = Worker(pool)
    a = pool.acquire_attention(key(), "a")
    original_page = pool.view(a).page_id
    ticket = pool.begin_offload(a)
    pool.submit(ticket)
    pool.release(a)
    b = pool.acquire_private("b")
    assert pool.begin_offload(b) is None
    if revive:
        a = pool.acquire_attention(key(), "new-owner")
        assert pool.view(a).page_id == original_page
    worker.finish(ticket, b"old")
    if revive:
        worker.run(pool.begin_restore(a), b"old")
        assert pool.begin_offload(b) is None
        pool.release(a)
    new_ticket = pool.begin_offload(b)
    worker.run(new_ticket, b"new")
    with pytest.raises(ValueError, match="finished"):
        pool.complete(ticket)
    worker.run(pool.begin_restore(b), b"new")
    pool.check_invariants()


@pytest.mark.parametrize("restore_first", [False, True])
def test_writeback_and_restore_pin_sources_through_last_owner_release(restore_first):
    pool = ImmutableKVPagePool(1, 1)
    worker = Worker(pool)
    lease = pool.acquire_attention(key(), "a")
    worker.store(lease, b"stable", disk=False)
    writeback = pool.begin_writeback(lease)
    restore = pool.begin_restore(lease)
    pool.submit(writeback)
    pool.submit(restore)
    pool.release(lease)
    other = pool.acquire_private("other")
    assert pool.begin_offload(other) is None
    first, second = (restore, writeback) if restore_first else (writeback, restore)
    worker.finish(first, b"stable")
    assert pool.begin_offload(other) is None
    worker.finish(second, b"stable")
    worker.store(other, b"other")
    pool.check_invariants()


def test_lru_demotion_and_shared_promotion_preserve_bytes():
    pool = ImmutableKVPagePool(2, 2)
    worker = Worker(pool)
    a = pool.acquire_attention(key(1), "a")
    b = pool.acquire_attention(key(2), "b")
    worker.store(a, b"A")
    worker.store(b, b"B")
    worker.run(pool.begin_restore(a), b"A")  # b is coldest
    c = pool.acquire_private("c")
    worker.store(c, b"C", disk=False)  # demotes b, not a
    assert pool.view(a).host_ready
    assert not pool.view(b).host_ready and pool.view(b).disk_ready
    alias = pool.acquire_attention(key(2), "another-b")
    pool.release(c)
    promotion = pool.begin_promotion(alias)
    assert pool.begin_promotion(b) is None  # exactly one producer
    assert pool.begin_restore(b) is None
    worker.run(promotion, b"B")
    assert pool.view(alias).host_slot == pool.view(b).host_slot
    worker.run(pool.begin_restore(b), b"B")
    pool.check_invariants()


def test_promotion_pins_disk_and_failed_promotion_keeps_disk_bytes():
    pool = ImmutableKVPagePool(2, 1)
    worker = Worker(pool)
    a = pool.acquire_attention(key(), "a")
    worker.store(a, b"A")
    assert pool.evict_host(a)
    b = pool.acquire_private("b")
    worker.store(b, b"B", disk=False)
    promotion = pool.begin_promotion(a)
    pool.submit(promotion)
    assert not pool.evict_disk(a)
    assert pool.begin_writeback(b) is None
    worker.finish(promotion, b"A", success=False)
    assert pool.view(a).disk_ready and not pool.view(a).host_ready
    worker.run(pool.begin_promotion(a), b"A")
    worker.run(pool.begin_writeback(b), b"B")  # can now evict a's disk copy
    assert pool.view(a).host_ready and not pool.view(a).disk_ready
    worker.run(pool.begin_restore(a), b"A")
    pool.check_invariants()


@pytest.mark.parametrize("pages,busy", [(3, 1), (5, 1), (8, 3)])
def test_partial_promotion_reservations_cancel_without_duplicate_free_slots(
    pages, busy
):
    pool = ImmutableKVPagePool(pages, pages)
    worker = Worker(pool)
    leases = [pool.acquire_attention(key(i), "a") for i in range(pages)]
    for i, lease in enumerate(leases):
        worker.store(lease, str(i).encode())
        assert pool.evict_host(lease)
    busy_leases = [pool.acquire_private("busy") for _ in range(busy)]
    for lease in busy_leases:
        pool.submit(pool.begin_offload(lease))
    promotions = [pool.begin_promotion(x) for x in leases]
    assert sum(t is not None for t in promotions) == pages - busy
    for ticket in promotions:
        if ticket is not None:
            pool.cancel(ticket)
    assert pool.stats()["host_used"] == busy
    assert all(pool.view(x).disk_ready for x in leases)
    allocated = []
    for _ in range(pages - busy):
        new = pool.acquire_private("new")
        ticket = pool.begin_offload(new)
        assert ticket is not None
        allocated.append(ticket.destination)
    assert len(set(allocated)) == pages - busy
    pool.check_invariants()


def test_two_phase_tickets_reject_early_completion_and_late_cancellation():
    pool = ImmutableKVPagePool(1)
    lease = pool.acquire_private("a")
    ticket = pool.begin_offload(lease)
    with pytest.raises(ValueError, match="unsubmitted"):
        pool.complete(ticket)
    pool.submit(ticket)
    with pytest.raises(ValueError, match="already submitted"):
        pool.submit(ticket)
    with pytest.raises(ValueError, match="must finish"):
        pool.cancel(ticket)
    pool.complete(ticket, success=False)
    assert not pool.view(lease).host_ready
    ticket = pool.begin_offload(lease)
    pool.cancel(ticket)
    with pytest.raises(ValueError, match="finished"):
        pool.cancel(ticket)
    pool.check_invariants()


@pytest.mark.parametrize("operation", ["writeback", "restore"])
def test_failed_reads_or_writeback_do_not_invalidate_the_host_source(operation):
    pool = ImmutableKVPagePool(1, 1)
    worker = Worker(pool)
    lease = pool.acquire_attention(key(), "a")
    worker.store(lease, b"valid-source", disk=False)
    ticket = getattr(pool, f"begin_{operation}")(lease)
    pool.submit(ticket)
    worker.finish(ticket, b"valid-source", success=False)
    assert pool.view(lease).host_ready
    assert not pool.view(lease).disk_ready
    worker.run(pool.begin_restore(lease), b"valid-source")
    pool.check_invariants()


def test_reserved_read_pins_before_submission_and_never_drops_the_last_copy():
    pool = ImmutableKVPagePool(1, 1)
    worker = Worker(pool)
    lease = pool.acquire_attention(key(), "a")
    worker.store(lease, b"A")
    read = pool.begin_restore(lease)
    assert not pool.evict_host(lease)  # protected before metadata is submitted
    assert pool.evict_disk(lease)
    pool.cancel(read)
    assert not pool.evict_host(lease)  # only ready copy left
    other = pool.acquire_private("b")
    assert pool.begin_offload(other) is None
    worker.run(pool.begin_restore(lease), b"A")
    pool.check_invariants()


def test_foreign_and_forged_handles_cannot_release_or_complete_owned_storage():
    pool = ImmutableKVPagePool(1)
    other = ImmutableKVPagePool(1)
    a = pool.acquire_private("a")
    b = other.acquire_private("b")
    assert a.lease_id == b.lease_id
    assert not pool.release(b)
    assert not pool.release(replace(a))
    with pytest.raises(ValueError, match="foreign"):
        pool.view(b)
    ticket = pool.begin_offload(a)
    foreign = other.begin_offload(b)
    assert ticket.ticket_id == foreign.ticket_id
    for wrong in (foreign, replace(ticket)):
        with pytest.raises(ValueError, match="foreign"):
            pool.submit(wrong)
    pool.check_invariants()


@pytest.mark.parametrize("seed", range(8))
def test_random_delayed_completion_and_slot_reuse_preserve_bytes(seed):
    rng = random.Random(seed)
    pool = ImmutableKVPagePool(5, 4)
    worker = Worker(pool)
    leases = []
    tickets = []
    begin = [
        pool.begin_offload,
        pool.begin_writeback,
        pool.begin_promotion,
        pool.begin_restore,
    ]
    for _ in range(600):
        choice = rng.randrange(6)
        if choice == 0 or not leases:
            lease = (
                pool.acquire_private("private")
                if rng.randrange(4) == 0
                else pool.acquire_attention(
                    key(rng.randrange(6), rng.randrange(2)), "owner"
                )
            )
            leases.append(lease)
        elif choice == 1:
            pool.release(leases.pop(rng.randrange(len(leases))))
        elif choice in (2, 3):
            lease = rng.choice(leases)
            value = str(pool.view(lease).page_id).encode()
            ticket = rng.choice(begin)(lease)
            if ticket is not None:
                submitted = bool(rng.randrange(2))
                if submitted:
                    pool.submit(ticket)
                tickets.append((ticket, value, submitted))
        elif choice == 4 and tickets:
            ticket, value, submitted = tickets.pop(rng.randrange(len(tickets)))
            if not submitted and rng.randrange(2):
                pool.cancel(ticket)
            else:
                if not submitted:
                    pool.submit(ticket)
                worker.finish(ticket, value, success=rng.randrange(5) != 0)
        else:
            lease = rng.choice(leases)
            (pool.evict_host if rng.randrange(2) else pool.evict_disk)(lease)
        pool.check_invariants()
    for lease in leases:
        pool.release(lease)
    for ticket, value, submitted in tickets:
        if submitted:
            worker.finish(ticket, value)
        else:
            pool.cancel(ticket)
        pool.check_invariants()
    assert pool.stats()["pages"] == 0
    assert pool.stats()["host_used"] == pool.stats()["disk_used"] == 0


@pytest.mark.parametrize("families", [1, 8, 16, 32])
def test_profile_sized_replay_inventory_keeps_shared_prefixes_and_private_tails(
    families,
):
    """Representative bytes and real slot counts; no model/GPU allocation.

    Each of eight rounds contains distinct prefix families. Only the same
    family's full attention pages repeat. Four private tails per owner
    stay distinct. This does not implement trajectory resumability checks.
    """
    pool = ImmutableKVPagePool(11915, 42366)
    worker = Worker(pool)
    trajectories = []
    copies = 0
    for round_id in range(8):
        for family in range(families):
            owner = f"{round_id}:{family}"
            trajectory = []
            for logical in range(216):
                for gid in [0, 1] if (logical + 1) % 8 == 0 else [0]:
                    value = f"{family}:{logical}:{gid}".encode()
                    lease = pool.acquire_attention(AttentionPageKey(gid, value), owner)
                    trajectory.append((lease, value))
                    if not pool.view(lease).host_ready:
                        worker.store(lease, value)
                        copies += 1
            for gid in range(4):
                lease = pool.acquire_private(owner)
                value = f"tail:{owner}:{gid}".encode()
                worker.store(lease, value)
                trajectory.append((lease, value))
            trajectories.append(trajectory)
        pool.check_invariants()
    assert copies == families * 243
    expected_slots = families * 243 + families * 8 * 4
    assert pool.stats()["host_used"] == expected_slots
    assert pool.stats()["disk_used"] == expected_slots
    # Removing a whole round frees only its private tails, not shared bytes.
    for trajectory in trajectories[:families]:
        for lease, _ in trajectory:
            pool.release(lease)
    assert pool.stats()["host_used"] == expected_slots - families * 4
    for trajectory in trajectories[families:]:
        for lease, value in trajectory:
            worker.run(pool.begin_restore(lease), value)
            pool.release(lease)
    pool.check_invariants()
    assert pool.stats()["pages"] == 0


@pytest.mark.parametrize("host,disk", [(0, 0), (-1, 0), (1, -1), (True, 0), (1, 1.5)])
def test_invalid_capacities(host, disk):
    with pytest.raises(ValueError):
        ImmutableKVPagePool(host, disk)


@pytest.mark.parametrize(
    "group,block_hash", [(-1, b"x"), (0, b""), (0, bytearray(b"x"))]
)
def test_invalid_attention_keys(group, block_hash):
    with pytest.raises(ValueError):
        AttentionPageKey(group, block_hash)
