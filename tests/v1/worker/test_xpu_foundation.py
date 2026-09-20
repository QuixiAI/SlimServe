# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for XPU spawning and graph breaks; no device allocations."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm import envs
from vllm.compilation import breakable_cudagraph as graphs
from vllm.platforms import xpu_affinity as affinity


@pytest.fixture
def config(monkeypatch):
    import vllm.platforms
    import vllm.utils.system_utils

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    monkeypatch.setattr(
        vllm.utils.system_utils,
        "get_mp_context",
        lambda: SimpleNamespace(get_start_method=lambda: "spawn"),
    )
    monkeypatch.setattr(envs, "VLLM_XPU_PER_WORKER_AFFINITY", True)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            nnodes_within_dp=1,
            data_parallel_rank_local=0,
            data_parallel_index=0,
            pipeline_parallel_size=1,
            tensor_parallel_size=4,
        )
    )


def test_affinity_indexes_parent_mask_and_restores_after_failure(config, monkeypatch):
    monkeypatch.setenv("ZE_AFFINITY_MASK", "3, 1, 2, 0")
    monkeypatch.delenv(affinity.XPU_WORKER_AFFINITY_PINNED_ENV, raising=False)
    with (
        pytest.raises(RuntimeError, match="spawn failure"),
        affinity.xpu_worker_affinity_env(1, config),
    ):
        assert os.environ["ZE_AFFINITY_MASK"] == "1"
        assert affinity.xpu_worker_affinity_pinned()
        raise RuntimeError("spawn failure")
    assert os.environ["ZE_AFFINITY_MASK"] == "3, 1, 2, 0"
    assert not affinity.xpu_worker_affinity_pinned()


def test_affinity_honors_dp_offset_and_rejects_unavailable_device(config, monkeypatch):
    config.parallel_config.data_parallel_rank_local = 1
    assert affinity.xpu_dp_adjusted_local_rank(config.parallel_config, 2) == 6
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0,1,2,3")
    with (
        pytest.raises(RuntimeError, match="outside"),
        affinity.xpu_worker_affinity_env(2, config),
    ):
        pytest.fail("invalid device must never spawn")
    assert os.environ["ZE_AFFINITY_MASK"] == "0,1,2,3"


def test_collective_replay_keeps_capture_tensor_address():
    capture = graphs.BreakableCUDAGraphCapture.__new__(graphs.BreakableCUDAGraphCapture)
    capture.segments = []
    capture._num_eager_breaks = 0
    capture._end_segment = Mock()
    capture._begin_segment = Mock()
    values = iter([torch.tensor([1.0, 2.0]), torch.tensor([4.0, 5.0])])
    result = capture.add_eager_tensor_output(lambda: next(values))
    pointer = result.data_ptr()
    capture.segments[0]()
    assert result.data_ptr() == pointer
    torch.testing.assert_close(result, torch.tensor([4.0, 5.0]))
    assert capture._num_eager_breaks == 1
    capture._end_segment.assert_called_once()
    capture._begin_segment.assert_called_once()


def test_collective_capture_rejects_non_tensor_output():
    capture = graphs.BreakableCUDAGraphCapture.__new__(graphs.BreakableCUDAGraphCapture)
    capture._end_segment = Mock()
    with pytest.raises(TypeError, match="one tensor"):
        capture.add_eager_tensor_output(lambda: None)


def test_xpu_graph_context_restores_stream_after_failure(monkeypatch):
    from vllm.v1.worker.xpu_model_runner import _xpu_graph

    old_stream, new_stream = object(), object()
    monkeypatch.setattr(torch.xpu, "current_stream", lambda: old_stream)
    set_stream = Mock()
    monkeypatch.setattr(torch.xpu, "set_stream", set_stream)
    graph = Mock()
    with (
        pytest.raises(RuntimeError, match="capture failure"),
        _xpu_graph(graph, pool="pool", stream=new_stream),
    ):
        raise RuntimeError("capture failure")
    graph.capture_begin.assert_called_once_with(pool="pool")
    graph.capture_end.assert_called_once()
    assert [call.args[0] for call in set_stream.call_args_list] == [
        new_stream,
        old_stream,
    ]


def test_xpu_environment_controls_are_registered(monkeypatch):
    for key in [
        "VLLM_XPU_PER_WORKER_AFFINITY",
        "VLLM_XPU_WORKER_AFFINITY_PINNED",
        "VLLM_XPU_GEMMA_NORM_FUSED",
        "VLLM_XPU_GRAPH_REPLAY_ORDER",
        "VLLM_XPU_FORCE_PIECEWISE_TP",
    ]:
        assert key in envs.environment_variables
    monkeypatch.setenv("VLLM_XPU_GRAPH_REPLAY_ORDER", "invalid")
    with pytest.raises(ValueError):
        envs.environment_variables["VLLM_XPU_GRAPH_REPLAY_ORDER"]()
