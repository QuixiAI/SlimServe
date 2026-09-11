# SPDX-License-Identifier: Apache-2.0
"""CPU index-level accounting of repeated immutable prefixes, not serving TPS.

Keep request lineages and tail states independent. This measures potential
attention-page sharing; it does not implement reference-counted sharing or
prove that changing the connector is safe.
"""

import argparse
import json

from vllm.v1.core.immutable_kv_pages import AttentionPageKey, ImmutableKVPagePool
from vllm.v1.core.kv_tier_index import HostKVTierIndex


def measure(owners=4, blocks=224, host_slots=512, disk_slots=2048, prefix_families=1):
    if owners < 1 or blocks < 8 or blocks % 8:
        raise ValueError(
            "positive owners and a positive multiple-of-eight prefix required"
        )
    if prefix_families < 1 or prefix_families > owners:
        raise ValueError("prefix_families must be between one and owners")
    index = HostKVTierIndex(
        num_slots=host_slots,
        num_disk_slots=disk_slots,
        attn_gids=[0, 1],
        attn_ratio={0: 1, 1: 8},
    )
    copied_attention = 0
    copied_tail = 0
    disk_writes = 0
    unique_attention = set()
    snapshots = []
    for owner_id in range(owners):
        owner = f"request-{owner_id}"
        # Distinct requests within a serving batch need not share prefixes.
        # Repeated benchmark rounds do reuse each request's prompt, however.
        # Model those independent families without claiming real token hashes.
        family = owner_id % prefix_families
        hash_base = family * blocks
        host = []
        for logical in range(blocks):
            block_hash = (hash_base + logical + 1).to_bytes(16, "little")
            for gid in sorted(index.due(logical)):
                slot = index.stage_attention(owner, logical, block_hash, gid=gid)
                assert slot is not None
                host.append(slot)
                copied_attention += 1
                unique_attention.add((gid, block_hash))
        tails = index.stage_tail_states(
            owner, blocks, 4, boundary_hash=(hash_base + blocks).to_bytes(16, "little")
        )
        assert tails is not None
        copied_tail += len(tails)
        host.extend(tails.values())
        index.confirm_writes(host)
        writes = index.take_disk_writes(host)
        disk_writes += len(writes)
        index.confirm_disk_writes(writes)
        snapshots.append(index.stats())
    return dict(
        scope="CPU index accounting only; no GPU timing or dedup implementation",
        implementation="exclusive-index",
        owners=owners,
        synthetic_prefix_families=prefix_families,
        owners_are_sequential_lineages_not_a_serving_concurrency_test=True,
        configured_host_slots=host_slots,
        configured_disk_slots=disk_slots,
        prefix_tokens=blocks * 576,
        attention_page_copies=copied_attention,
        distinct_attention_pages=len(unique_attention),
        tail_page_copies=copied_tail,
        disk_page_writes=disk_writes,
        avoidable_attention_copies_if_safely_shared=copied_attention
        - len(unique_attention),
        snapshots=snapshots,
    )


def measure_shared(
    owners=4, blocks=224, host_slots=512, disk_slots=2048, prefix_families=1
):
    """Exercise the new physical pool, without pretending it is a connector.

    Copies complete synchronously here; delayed IO and representative-byte
    fidelity are tested in test_immutable_kv_pages. This accounting retains
    every lease and does not implement trajectory LRU or hash-chain lookup.
    """
    if owners < 1 or blocks < 8 or blocks % 8:
        raise ValueError(
            "positive owners and a positive multiple-of-eight prefix required"
        )
    if prefix_families < 1 or prefix_families > owners:
        raise ValueError("prefix_families must be between one and owners")
    pool = ImmutableKVPagePool(host_slots, disk_slots)
    trajectories = []
    attention_copies = tail_copies = disk_writes = 0

    def store(lease):
        nonlocal disk_writes
        copied = False
        view = pool.view(lease)
        if not (view.host_ready or view.disk_ready):
            ticket = pool.begin_offload(lease)
            if ticket is None:
                raise ValueError(
                    "inventory exceeds capacity; no trajectory eviction modeled"
                )
            pool.submit(ticket)
            pool.complete(ticket)
            copied = True
        ticket = pool.begin_writeback(lease)
        if ticket is not None:
            pool.submit(ticket)
            pool.complete(ticket)
            disk_writes += 1
        return copied

    for owner_id in range(owners):
        owner = f"request-{owner_id}"
        family = owner_id % prefix_families
        trajectory = []
        for logical in range(blocks):
            block_hash = (family * blocks + logical + 1).to_bytes(16, "little")
            for gid in [0, 1] if (logical + 1) % 8 == 0 else [0]:
                lease = pool.acquire_attention(AttentionPageKey(gid, block_hash), owner)
                trajectory.append(lease)
                attention_copies += int(store(lease))
        for _ in range(4):
            lease = pool.acquire_private(owner)
            trajectory.append(lease)
            tail_copies += int(store(lease))
        trajectories.append(trajectory)
    pool.check_invariants()
    return dict(
        scope="CPU physical-pool accounting; not trajectory lookup, DMA or serving TPS",
        implementation="shared-pool",
        owners=owners,
        synthetic_prefix_families=prefix_families,
        owners_are_sequential_lineages_not_a_serving_concurrency_test=True,
        configured_host_slots=host_slots,
        configured_disk_slots=disk_slots,
        prefix_tokens=blocks * 576,
        attention_page_copies=attention_copies,
        tail_page_copies=tail_copies,
        disk_page_writes=disk_writes,
        owners_with_all_pages_ready=sum(
            all(
                pool.view(lease).host_ready or pool.view(lease).disk_ready
                for lease in t
            )
            for t in trajectories
        ),
        final_pool_stats=pool.stats(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owners", type=int, nargs="+", default=[4])
    parser.add_argument("--blocks", type=int, default=224)
    parser.add_argument("--prefix-families", type=int, default=1)
    parser.add_argument(
        "--implementation",
        choices=["exclusive-index", "shared-pool"],
        default="exclusive-index",
    )
    parser.add_argument("--host-slots", type=int, default=512)
    parser.add_argument("--disk-slots", type=int, default=2048)
    args = parser.parse_args()
    implementation = measure_shared if args.implementation == "shared-pool" else measure
    for owners in args.owners:
        print(
            json.dumps(
                implementation(
                    owners=owners,
                    blocks=args.blocks,
                    prefix_families=args.prefix_families,
                    host_slots=args.host_slots,
                    disk_slots=args.disk_slots,
                )
            ),
            flush=True,
        )
