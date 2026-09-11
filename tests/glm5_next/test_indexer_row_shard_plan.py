# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of rank ownership, padding, and the collective row order."""

import pytest

from benchmarks.benchmark_glm5_next_indexer_row_shard import row_shard


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 16, 31, 32, 64])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_rank_order_gather_recovers_each_row_once(rows, world):
    chunks = []
    covered = []
    for rank in range(world):
        local, start, stop, padded = row_shard(rows, world, rank)
        assert padded == local * world
        valid = list(range(start, stop))
        covered.extend(valid)
        chunks.extend(valid + [-1] * (local - len(valid)))
    assert covered == list(range(rows))
    assert chunks[:rows] == list(range(rows))
    assert all(value == -1 for value in chunks[rows:])


@pytest.mark.parametrize(
    "rows,world,rank", [(0, 8, 0), (8, 0, 0), (8, 8, -1), (8, 8, 8)]
)
def test_invalid_plan_rejected(rows, world, rank):
    with pytest.raises(ValueError, match="in-range rank"):
        row_shard(rows, world, rank)
