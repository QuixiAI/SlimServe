# SPDX-License-Identifier: Apache-2.0
"""CPU fault injection for the benchmark's graph-output acceptance gate."""

from contextlib import contextmanager
import hashlib

import pytest
import torch

from benchmarks import benchmark_topk_sample as bench
from benchmarks.kernels.replay_glm53_moe_up import sha


@pytest.mark.parametrize("corrupt_replay", [False, True])
def test_graph_output_is_checked_before_timing(monkeypatch, corrupt_replay):
    state = {"capturing": False, "events": 0}

    class Graph:
        def replay(self):
            state["output"].fill_(8 if corrupt_replay else 7)

    @contextmanager
    def capture(graph):
        state["capturing"] = True
        try:
            yield
        finally:
            state["capturing"] = False

    class Event:
        def __init__(self, **kwargs):
            state["events"] += 1

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 2.5

    def operation():
        output = torch.tensor([7])
        if state["capturing"]:
            state["output"] = output
        return output

    monkeypatch.setattr(bench.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench.torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(bench.torch.cuda, "graph", capture)
    monkeypatch.setattr(bench.torch.cuda, "Event", Event)
    if corrupt_replay:
        with pytest.raises(AssertionError, match="Graph output differs"):
            bench.measure(operation, 2)
        assert state["events"] == 0
    else:
        result = bench.measure(operation, 2)
        assert result["graph_correct"]
        assert result["graph_cuda_us"] == [1250.0] * 3


def test_replay_digest_does_not_require_python311_file_digest(monkeypatch, tmp_path):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    content = b"quantized-weights" * 100000
    path = tmp_path / "weights.bin"
    path.write_bytes(content)
    assert sha(path) == hashlib.sha256(content).hexdigest()
