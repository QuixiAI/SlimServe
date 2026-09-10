# SPDX-License-Identifier: Apache-2.0
"""CPU plumbing gates only; no claim of GPU collective or arithmetic parity."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.distributed as distributed
from benchmarks import glm5_next_indexer_gather_candidate as candidate
from vllm.model_executor.layers import glm5_next_indexer as serving


def inputs(rows):
    return (
        torch.zeros(rows, 32, 128, dtype=torch.bfloat16),
        torch.zeros(rows, 32), torch.zeros(4, 128),
        torch.zeros(2, 576, 64, dtype=torch.bfloat16),
        torch.zeros(7, 2, dtype=torch.int32),
        torch.arange(rows, dtype=torch.int32) % 7,
        torch.arange(rows, dtype=torch.int32) + 1000,
        torch.empty(rows, 512), 512, 576, 128**-0.5, 512,
        torch.empty(rows, 2080, dtype=torch.int32), 4,
    )


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
def test_disabled_preserves_serving_fallback(monkeypatch, rows):
    original = Mock()
    monkeypatch.setattr(candidate, "_pooled_select", original)
    monkeypatch.setattr(candidate, "get_tp_group", Mock(side_effect=AssertionError))
    args = inputs(rows)
    candidate.pooled_select_existing_tp(*args, row_shard=False)
    assert original.call_count == 1
    assert all(actual is expected for actual, expected
               in zip(original.call_args.args, args))
    assert original.call_args.kwargs == {"adaptive_score": False}


@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("production", [False, True])
def test_owned_communicator_order_and_global_rows(monkeypatch, rows, rank, production):
    args = inputs(rows)
    events = []
    local_rows = rows // 8
    lo, hi = rank * local_rows, (rank + 1) * local_rows
    expected = torch.arange(rows, dtype=torch.int32)[:, None].expand(rows, 512)

    def topk(*call):
        events.append("topk")
        assert call[4] is args[4]  # Full block table, not a rank-local table.
        torch.testing.assert_close(call[5], args[5][lo:hi])
        torch.testing.assert_close(call[6], args[6][lo:hi])
        assert call[0].shape[0] == local_rows
        assert call[7].data_ptr() == args[7].data_ptr()
        return expected[lo:hi].contiguous()

    def gather(out, local):
        events.append("gather")
        torch.testing.assert_close(local, expected[lo:hi])
        out.copy_(expected)

    class Expand:
        def __getitem__(self, grid):
            assert grid == (rows,)

            def launch(sel, visible, output, stride, **kwargs):
                events.append("expand")
                torch.testing.assert_close(sel, expected)
                assert visible is args[6] and output is args[12]
                assert stride == 512 and kwargs["OUT_W"] == 2080

            return launch

    communicator = SimpleNamespace(
        wait_for_comm_init=lambda: events.append("ready"),
        pynccl_comm=SimpleNamespace(disabled=False, all_gather=gather),
    )
    group = SimpleNamespace(world_size=8, rank_in_group=rank,
                            device_communicator=communicator)
    module = serving if production else candidate
    monkeypatch.setattr(distributed if production else candidate,
                        "get_tp_group", lambda: group)
    monkeypatch.setattr(module, "_pooled_topk", topk)
    monkeypatch.setattr(module, "_expand_topk_kernel", Expand())
    selector = (serving._pooled_select if production
                else candidate.pooled_select_existing_tp)
    selector(*args, row_shard=True)
    assert events == ["ready", "topk", "gather", "expand"]


@pytest.mark.parametrize("pynccl", [None, SimpleNamespace(disabled=True)])
@pytest.mark.parametrize("production", [False, True])
def test_missing_communicator_fails_closed(monkeypatch, pynccl, production):
    group = SimpleNamespace(world_size=8, rank_in_group=0,
                            device_communicator=SimpleNamespace(
                                wait_for_comm_init=lambda: None,
                                pynccl_comm=pynccl))
    monkeypatch.setattr(distributed if production else candidate,
                        "get_tp_group", lambda: group)
    selector = (serving._pooled_select if production
                else candidate.pooled_select_existing_tp)
    with pytest.raises(AssertionError):
        selector(*inputs(16), row_shard=True)
