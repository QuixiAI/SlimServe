# SPDX-License-Identifier: Apache-2.0
"""All-rank metadata/lifetime tests, not GPU or connector validation."""

import pytest

from vllm.v1.core.immutable_kv_pages import (
    ImmutableKVPagePool,
    PageCompletionBarrier,
)


def reserve(pool, count=1):
    leases = [pool.acquire_private(str(i)) for i in range(count)]
    return leases, [pool.begin_offload(lease) for lease in leases]


def test_duplicate_rank_acks_cannot_make_a_page_ready_early():
    pool = ImmutableKVPagePool(2)
    leases, tickets = reserve(pool, 2)
    barrier = PageCompletionBarrier(pool, 8)
    batch = barrier.submit(tickets)
    for rank in range(7):
        assert not barrier.acknowledge(batch, rank)
        assert not barrier.acknowledge(batch, rank)
        assert all(not pool.view(lease).host_ready for lease in leases)
    assert barrier.pending() == 1
    assert barrier.acknowledge(batch, 7)
    assert all(pool.view(lease).host_ready for lease in leases)
    assert barrier.pending() == 0
    assert not barrier.acknowledge(batch, 7)
    pool.check_invariants()


@pytest.mark.parametrize("failed_rank", [0, 3, 7])
def test_failure_keeps_cancelled_owners_buffers_pinned_until_every_rank_finishes(
    failed_rank,
):
    pool = ImmutableKVPagePool(1)
    leases, tickets = reserve(pool)
    barrier = PageCompletionBarrier(pool, 8)
    batch = barrier.submit(tickets)
    pool.release(leases[0])
    new = pool.acquire_private("new")
    for rank in range(7):
        assert not barrier.acknowledge(batch, rank, success=rank != failed_rank)
        assert pool.begin_offload(new) is None
    assert barrier.acknowledge(batch, 7, success=failed_rank != 7)
    assert pool.begin_offload(new) is not None
    pool.check_invariants()


def test_failed_writeback_preserves_host_copy_and_releases_disk_reservation():
    pool = ImmutableKVPagePool(1, 1)
    leases, tickets = reserve(pool)
    barrier = PageCompletionBarrier(pool, 2)
    batch = barrier.submit(tickets)
    barrier.acknowledge(batch, 0)
    barrier.acknowledge(batch, 1)
    writeback = pool.begin_writeback(leases[0])
    batch = barrier.submit([writeback])
    assert not barrier.acknowledge(batch, 0, success=False)
    assert pool.view(leases[0]).disk_slot is not None
    assert barrier.acknowledge(batch, 1)
    assert pool.view(leases[0]).host_ready
    assert pool.view(leases[0]).disk_slot is None
    pool.check_invariants()


@pytest.mark.parametrize("method", ["submit_all", "cancel_all"])
def test_batch_validation_is_atomic_when_one_ticket_is_already_submitted(method):
    pool = ImmutableKVPagePool(2)
    _, tickets = reserve(pool, 2)
    pool.submit(tickets[1])
    with pytest.raises(ValueError, match="submission state"):
        getattr(pool, method)(tickets)
    pool.cancel(tickets[0])  # first ticket was not partially submitted/cancelled
    pool.complete(tickets[1], success=False)
    assert pool.stats()["host_used"] == 0
    pool.check_invariants()


def test_complete_batch_validation_is_atomic():
    pool = ImmutableKVPagePool(2)
    leases, tickets = reserve(pool, 2)
    pool.submit(tickets[0])
    with pytest.raises(ValueError, match="submission state"):
        pool.complete_all(tickets, success=True)
    assert not pool.view(leases[0]).host_ready
    pool.cancel(tickets[1])
    pool.complete(tickets[0])
    pool.check_invariants()


def test_duplicate_tickets_rejected_before_any_submission():
    pool = ImmutableKVPagePool(1)
    _, tickets = reserve(pool)
    barrier = PageCompletionBarrier(pool, 8)
    with pytest.raises(ValueError, match="distinct"):
        barrier.submit(tickets * 2)
    assert barrier.pending() == 0
    pool.cancel(tickets[0])
    pool.check_invariants()


def test_reserved_batch_cancellation_releases_each_slot_exactly_once():
    pool = ImmutableKVPagePool(4)
    _, tickets = reserve(pool, 4)
    pool.cancel_all(tickets)
    assert pool.stats()["host_used"] == 0
    pool.check_invariants()
    with pytest.raises(ValueError, match="finished"):
        pool.cancel_all(tickets)
    _, new = reserve(pool, 4)
    assert len({ticket.destination for ticket in new}) == 4
    pool.check_invariants()


def test_retired_batch_ids_do_not_complete_reused_slots():
    pool = ImmutableKVPagePool(1)
    barrier = PageCompletionBarrier(pool, 2)
    old_leases, old_tickets = reserve(pool)
    old = barrier.submit(old_tickets)
    barrier.acknowledge(old, 0)
    barrier.acknowledge(old, 1)
    pool.release(old_leases[0])
    new_leases, new_tickets = reserve(pool)
    new = barrier.submit(new_tickets)
    assert new > old
    assert not barrier.acknowledge(old, 0)
    assert not barrier.acknowledge(old, 1)
    assert not pool.view(new_leases[0]).host_ready
    barrier.acknowledge(new, 1)
    assert barrier.acknowledge(new, 0)
    pool.check_invariants()


def test_conflicting_ack_is_not_silently_counted_twice():
    pool = ImmutableKVPagePool(1)
    _, tickets = reserve(pool)
    barrier = PageCompletionBarrier(pool, 2)
    batch = barrier.submit(tickets)
    barrier.acknowledge(batch, 0)
    with pytest.raises(ValueError, match="conflicting"):
        barrier.acknowledge(batch, 0, success=False)
    assert barrier.pending() == 1
    assert barrier.acknowledge(batch, 1)


@pytest.mark.parametrize("rank", [-1, 8, True])
def test_invalid_rank_does_not_change_pending_state(rank):
    pool = ImmutableKVPagePool(1)
    _, tickets = reserve(pool)
    barrier = PageCompletionBarrier(pool, 8)
    batch = barrier.submit(tickets)
    with pytest.raises(ValueError, match="rank"):
        barrier.acknowledge(batch, rank)
    assert barrier.pending() == 1
    pool.check_invariants()


@pytest.mark.parametrize("count", [0, -1, True])
def test_invalid_rank_count_is_rejected(count):
    with pytest.raises(ValueError, match="num_ranks"):
        PageCompletionBarrier(ImmutableKVPagePool(1), count)
