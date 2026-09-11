# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from benchmarks.kernels import benchmark_glm53_indexer_shard as probe


@pytest.mark.parametrize("change", ["order", "region-swap", "tail-order"])
def test_comparison_keeps_pool_sets_separate_from_exact_tail(monkeypatch, change):
    group = SimpleNamespace(world_size=1, cpu_group=object())

    def gather(output, local, *, group):
        output[:] = [local]

    monkeypatch.setattr(probe.dist, "all_gather_object", gather)
    a = torch.arange(2051, dtype=torch.int32)[None, :]
    b = a.clone()
    left, right = {
        "order": (0, 1),
        "region-swap": (0, 2048),
        "tail-order": (2048, 2049),
    }[change]
    b[:, [left, right]] = b[:, [right, left]]
    assert torch.equal(a.sort(1).values, b.sort(1).values)
    result = probe.compare_outputs(a, b, group)
    assert result["sets_and_tail_exact_all_ranks"] is (change == "order")
    assert not result["rank0_order_equal"]
    assert result["per_rank"][0]["tail_equal"] is (change == "order")


def test_comparison_records_other_rank_failure_and_rank0_order(monkeypatch):
    cpu_group = object()
    group = SimpleNamespace(world_size=2, cpu_group=cpu_group)
    remote = dict(pool_sets_equal=True, tail_equal=False, order_equal=False)

    def gather(output, local, *, group):
        assert group is cpu_group
        assert local == dict(pool_sets_equal=True, tail_equal=True, order_equal=True)
        output[:] = [remote, local]

    monkeypatch.setattr(probe.dist, "all_gather_object", gather)
    a = torch.arange(2051, dtype=torch.int32)[None, :]
    result = probe.compare_outputs(a, a.clone(), group)
    assert not result["sets_and_tail_exact_all_ranks"]
    assert not result["rank0_order_equal"]
    assert result["per_rank"][0] == remote
    assert result["per_rank"][1]["tail_equal"]
