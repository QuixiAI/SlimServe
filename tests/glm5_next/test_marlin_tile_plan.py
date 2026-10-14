# SPDX-License-Identifier: Apache-2.0
"""CPU-only benchmark fixtures; not Marlin kernel correctness tests."""

import pytest
import torch

from benchmarks.benchmark_glm5_next_marlin_tiles import (
    configurations,
    marlin_partition_plan,
    projection_configurations,
    route_bank,
    route_summary,
)


@pytest.mark.parametrize("rows", [1, 8, 16, 32])
@pytest.mark.parametrize("pattern", ["shared", "disjoint", "uniform"])
def test_routes_are_distinct_per_token_and_in_range(rows, pattern):
    ids = route_bank(rows, 16, pattern)
    assert ids.shape == (16, rows, 8) and ids.dtype == torch.int32
    assert ids.min() >= 0 and ids.max() < 288
    assert (ids.sort(-1).values.diff(dim=-1) > 0).all()
    torch.testing.assert_close(ids, route_bank(rows, 16, pattern), atol=0, rtol=0)
    assert not torch.equal(ids, route_bank(rows, 16, pattern, seed=54))
    assert ids.unique().numel() >= (80 if pattern == "uniform" else 128)
    summary = route_summary(ids[0])
    assert summary["real_rows"] == rows * 8
    assert summary["padded_rows"] % 8 == 0
    if pattern == "disjoint":
        assert summary["active_experts"] == rows * 8
        assert summary["padded_rows"] == rows * 64
    if pattern == "shared":
        assert summary["active_experts"] == 8
        assert summary["max_rows_per_expert"] == rows


def test_tuning_configs_match_generator_tile_families_and_shapes():
    configs = configurations()
    assert configs[0] == (-1, -1, -1)
    assert len(configs) == len(set(configs)) == 17
    for k, n, occupancy in configs[1:]:
        assert k * n // 64 in (128, 256)
        assert occupancy in (1, 2, 3, 4)
        for size_k, size_n in ((4096, 512), (256, 4096)):
            assert size_k % k == size_n % n == 0


def test_direct_kernel_is_opt_in_and_down_projection_only():
    assert projection_configurations("down") == configurations()
    assert projection_configurations("gate_up", True) == configurations()
    assert projection_configurations("down", True)[-2:] == [(0, 64, 0), (0, 128, 0)]
    with pytest.raises(ValueError):
        projection_configurations("bad", True)


@pytest.mark.parametrize(
    "padded,dp,tail,split",
    [
        (64, 0, 256, 256),  # c1: eight experts, current 432-CTA down launch
        (2048, 7776, 416, 0),  # c32 disjoint: native tail is also full-K
        (256, 864, 160, 160),  # c32 shared: 32 aligned expert blocks
        (0, 0, 0, 0),
    ],
)
def test_native_partition_counts_depend_on_routing(padded, dp, tail, split):
    plan = marlin_partition_plan(padded, 256, 4096, 64, 128, 4)
    assert plan["full_k_dp_tiles"] == dp
    assert plan["tail_tiles"] == tail
    assert plan["multi_cta_tiles"] == split
    assert plan["total_tiles"] == dp + tail


@pytest.mark.parametrize("padded", [8, 64, 256, 512, 1024, 2048])
def test_partition_plan_covers_tiles_for_every_explicit_tuning_choice(padded):
    for k, n, occupancy in configurations()[1:]:
        for size_k, size_n in ((4096, 512), (256, 4096)):
            plan = marlin_partition_plan(padded, size_k, size_n, k, n, occupancy)
            assert plan["total_tiles"] == plan["full_k_dp_tiles"] + plan["tail_tiles"]
            assert 0 <= plan["multi_cta_tiles"] <= plan["tail_tiles"]
            assert plan["tail_cta_fragments"] >= plan["tail_tiles"]


@pytest.mark.parametrize("args", [(2, 16, "shared"), (1, 0, "uniform"), (1, 1, "bad")])
def test_invalid_fixture_requests_fail(args):
    with pytest.raises(ValueError):
        route_bank(*args)
