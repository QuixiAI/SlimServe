# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU graph/GGUF knobs are envs.py variables honoured at their use sites.

Runs on CPU: the QuixiCore stream sync and the graph are stand-ins.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm
import vllm.envs as envs
from vllm.compilation import breakable_cudagraph as bcg


@pytest.mark.parametrize(
    "order, lead, trail",
    [
        (None, True, True),
        ("sync", True, True),
        ("LEAD", True, False),
        ("trail", False, True),
        ("none", False, False),
    ],
)
def test_replay_order_controls_waits(monkeypatch, order, lead, trail):
    if order is None:
        monkeypatch.delenv("VLLM_XPU_GRAPH_REPLAY_ORDER", raising=False)
    else:
        monkeypatch.setenv("VLLM_XPU_GRAPH_REPLAY_ORDER", order)
    sync = MagicMock()
    qc = SimpleNamespace(synchronize_current_stream=sync)
    monkeypatch.setitem(sys.modules, "vllm._quixicore_C", qc)
    monkeypatch.setattr(vllm, "_quixicore_C", qc, raising=False)
    # Skip the one-time torch.xpu.synchronize rebinding.
    monkeypatch.setattr(bcg, "_XPU_SYNC_BINDINGS_INSTALLED", True)
    graph = MagicMock()

    bcg._xpu_graph_replayer(graph)()

    graph.replay.assert_called_once_with()
    assert sync.called == lead
    assert graph.synchronize.called == trail


def test_replay_order_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("VLLM_XPU_GRAPH_REPLAY_ORDER", "sometimes")
    with pytest.raises(ValueError):
        envs.VLLM_XPU_GRAPH_REPLAY_ORDER  # noqa: B018


@pytest.mark.parametrize("value, enabled", [(None, False), ("1", True), ("on", True)])
def test_gguf_q8_split_is_an_envs_knob(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv("VLLM_XPU_GGUF_Q8_SPLIT", raising=False)
    else:
        monkeypatch.setenv("VLLM_XPU_GGUF_Q8_SPLIT", value)
    assert envs.VLLM_XPU_GGUF_Q8_SPLIT is enabled
