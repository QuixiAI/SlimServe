# SPDX-License-Identifier: Apache-2.0
"""Cached graphs carry a stable name, not a CUDA pointer from a dead process."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers import glm5_next_mhc_project as mhc


def context(stream):
    config = SimpleNamespace(static_forward_context={})
    mhc.register_projection_stream(config, "model.mhc_stream", stream)
    return ForwardContext(config.static_forward_context, None, {})


def test_each_forward_context_resolves_its_own_stream(monkeypatch):
    calls = []
    monkeypatch.setattr(mhc, "project_transition", lambda *a: calls.append(a[9]))
    tensor = torch.empty(1)
    for handle in (1234, 5678, 1234):
        stream = SimpleNamespace(device=tensor.device, cuda_stream=handle)
        with override_forward_context(context(stream)):
            mhc.project_transition_runtime(*([tensor] * 9), "model.mhc_stream", False)
    assert calls == [1234, 5678, 1234]


def test_duplicate_stream_registration_cannot_replace_live_owner():
    config = SimpleNamespace(static_forward_context={})
    stream = object()
    mhc.register_projection_stream(config, "model.mhc_stream", stream)
    with pytest.raises(ValueError, match="duplicate"):
        mhc.register_projection_stream(config, "model.mhc_stream", object())
    assert config.static_forward_context["model.mhc_stream"].stream is stream
    with pytest.raises(ValueError):
        mhc.register_projection_stream(config, "", stream)


@pytest.mark.parametrize("bad", ["missing", "type", "device"])
def test_invalid_owner_fails_before_touching_cuda(bad, monkeypatch):
    def unexpected(*args):
        raise AssertionError("must not launch")

    monkeypatch.setattr(mhc, "project_transition", unexpected)
    tensor = torch.empty(1)
    ctx = context(SimpleNamespace(device=torch.device("meta"), cuda_stream=1234))
    if bad == "missing":
        ctx.no_compile_layers.clear()
    elif bad == "type":
        ctx.no_compile_layers["model.mhc_stream"] = object()
    exception = {"missing": KeyError, "type": TypeError, "device": ValueError}[bad]
    with override_forward_context(ctx), pytest.raises(exception):
        mhc.project_transition_runtime(*([tensor] * 9), "model.mhc_stream", False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_inductor_disk_cache_restart_and_cuda_graph_replay(tmp_path):
    env = os.environ | {
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONFAULTHANDLER": "1",
    }
    reports = []
    for _ in range(2):
        run = subprocess.run(
            [sys.executable, "-m", "tests.glm5_next.test_mhc_runtime_stream"],
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        report = next(line for line in run.stdout.splitlines() if line.startswith("{"))
        reports.append(json.loads(report))
    assert reports[0]["pid"] != reports[1]["pid"]
    assert reports[1]["fxgraph_cache_hit"] >= 1, reports
    print(json.dumps({"restart_reports": reports}))
    generated = list((tmp_path / "inductor").rglob("*.py"))
    calls = [
        line
        for path in generated
        for line in path.read_text().splitlines()
        if "torch.ops.vllm.glm5_mhc_project_runtime.default(" in line
    ]
    assert calls and all("model.mhc_stream" in line for line in calls)
    for report in reports:
        assert all(str(report["handle"]) not in line for line in calls)


@torch.inference_mode()
def cache_worker():
    from torch._dynamo.utils import counters

    from tests.glm5_next.test_mhc_project_candidate import inputs

    torch.manual_seed(271)
    args = inputs(8, 3336, "cuda")
    # Hold distinct streams alive so this is not accidentally using the default.
    streams = [torch.cuda.Stream() for _ in range(3)]
    stream = streams[-1]

    def forward(*args):
        return torch.ops.vllm.glm5_mhc_project_runtime(
            args[0] + 1, *args[1:], "model.mhc_stream", False
        )

    compiled = torch.compile(forward, fullgraph=True)
    with override_forward_context(context(stream)):
        for _ in range(3):
            actual = compiled(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = compiled(*args)
        for _ in range(4):
            args[0].normal_()
            args[1].normal_()
            graph.replay()
            expected = mhc.project_transition(
                args[0] + 1, *args[1:], stream.cuda_stream, False
            )
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "handle": stream.cuda_stream,
                "fxgraph_cache_hit": counters["inductor"]["fxgraph_cache_hit"],
                "fxgraph_cache_miss": counters["inductor"]["fxgraph_cache_miss"],
                "replays": 4,
            }
        )
    )


if __name__ == "__main__":
    cache_worker()
