# SPDX-License-Identifier: Apache-2.0
"""Execute the real runner method with CPU request/projection/transfer fixtures."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slimserve.score_journal import STAGES, ScoreJournal
from tests.slimserve.test_prompt_score_chunks import CPUSampler, reference
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.prompt_logprobs import MODES


@pytest.fixture
def runner_method(monkeypatch):
    source = Path("vllm/v1/worker/gpu_model_runner.py").read_text()
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_prompt_logprobs_dict"
    )
    environment = SimpleNamespace(value="1")
    namespace = dict(
        torch=torch,
        LogprobsTensors=LogprobsTensors,
        os=SimpleNamespace(getenv=lambda name, default: environment.value),
        async_tensor_h2d=lambda ids, device: torch.tensor(ids, device=device),
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<runner-method>", "exec"),
        namespace,
    )
    monkeypatch.setattr(
        LogprobsTensors,
        "empty_cpu",
        staticmethod(
            lambda rows, cols: LogprobsTensors(
                torch.empty(rows, cols, dtype=torch.int32),
                torch.empty(rows, cols),
                torch.empty(rows, dtype=torch.int64),
            )
        ),
    )
    return namespace[method.name], environment


def make_runner(prompt_rows, count, mode, journal=None):
    ids = (torch.arange(prompt_rows + 1) % 17).tolist()
    request = SimpleNamespace(
        prompt_token_ids=ids,
        num_computed_tokens=0,
        in_progress_prompt_logprobs_cpu=None,
    )
    projections, syncs = [], []

    def project(hidden):
        projections.append(hidden.clone())
        return hidden

    runner = SimpleNamespace(
        num_prompt_logprobs={"a": count},
        _slimserve_score_journal=journal,
        requests={"a": request},
        device="cpu",
        model_config=SimpleNamespace(
            logprobs_mode=mode, hf_config=SimpleNamespace(model_type="glm5_next")
        ),
        model=SimpleNamespace(compute_logits=project),
        sampler=CPUSampler,
        input_batch=SimpleNamespace(req_id_to_index={"a": 0}),
        query_start_loc=SimpleNamespace(np=np.array([0])),
        _sync_device=lambda: syncs.append(True),
    )
    return runner, projections, syncs


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows", [639, 1024, 1025, 2051])
@pytest.mark.parametrize("count", [0, 5])
@pytest.mark.parametrize("chunk_rows", [0, 1024])
def test_full_request_keeps_one_projection_and_exact_scores(
    runner_method, mode, rows, count, chunk_rows
):
    method, env = runner_method
    env.value = "1" if chunk_rows else "0"
    runner, projections, syncs = make_runner(rows, count, mode)
    hidden = torch.randn(
        rows, 17, generator=torch.Generator().manual_seed(530910)
    ).bfloat16()
    expected = reference(
        hidden, torch.tensor(runner.requests["a"].prompt_token_ids[1:]), count, mode
    )
    actual = method(runner, hidden, {"a": rows + 1})["a"]
    assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
    assert len(projections) == 1 and torch.equal(projections[0], hidden)
    assert syncs == [True] and runner.num_prompt_logprobs == {}
    assert runner.requests["a"].in_progress_prompt_logprobs_cpu is None


def test_chunked_prefill_defers_completion_and_preserves_offsets(runner_method):
    method, _ = runner_method
    runner, projections, syncs = make_runner(2051, 5, "raw_logprobs")
    hidden = torch.randn(2051, 17).bfloat16()
    expected = reference(
        hidden,
        torch.tensor(runner.requests["a"].prompt_token_ids[1:]),
        5,
        "raw_logprobs",
    )
    assert method(runner, hidden[:1025], {"a": 1025}) == {}
    runner.requests["a"].num_computed_tokens = 1025
    assert method(runner, hidden[1025:], {"a": 1026}) == {}
    runner.requests["a"].num_computed_tokens = 2051
    actual = method(runner, hidden[:1], {"a": 1})["a"]
    assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
    assert [p.shape[0] for p in projections] == [1025, 1026]
    assert syncs == [True]


@pytest.mark.parametrize("rows", [639, 1024, 1025])
@pytest.mark.parametrize("enabled", [False, True])
def test_chunk_dispatch_coverage_not_inferred_from_score_equality(
    runner_method, monkeypatch, rows, enabled
):
    from vllm.v1.sample import prompt_logprobs

    method, env = runner_method
    env.value = "1" if enabled else "0"
    original = prompt_logprobs.gather_prompt_logprobs
    calls = []

    def observed(logits, *args, **kwargs):
        calls.append(logits.shape[0])
        return original(logits, *args, **kwargs)

    monkeypatch.setattr(prompt_logprobs, "gather_prompt_logprobs", observed)
    runner, projections, _ = make_runner(rows, 0, "raw_logprobs")
    method(runner, torch.zeros(rows, 17).bfloat16(), {"a": rows + 1})
    assert len(projections) == 1
    # The 512-prefix + 128-continuation quality workload has 639 scored rows:
    # it cannot exercise the memory fix even when the option is enabled.
    assert calls == ([rows] if enabled and rows > 1024 else [])


def test_empty_skipped_embedding_and_invalid_configuration(runner_method):
    method, env = runner_method
    runner, projections, syncs = make_runner(1025, 0, "raw_logprobs")
    hidden = torch.zeros(1025, 17)
    assert method(runner, hidden, {}) == {}
    runner.requests["a"].prompt_token_ids = None
    assert method(runner, hidden, {"a": 1026}) == {}
    assert not projections and not syncs
    env.value = "-1"
    with pytest.raises(ValueError, match="must be 0 or 1"):
        method(runner, hidden, {"a": 1026})
    runner.num_prompt_logprobs = {}
    assert method(runner, hidden, {}) == {}


def test_journal_stages_stay_exact_with_chunks_enabled(runner_method, tmp_path):
    method, env = runner_method
    env.value = "1"
    runner, _, _ = make_runner(639, 0, "raw_logprobs")
    path = tmp_path / "journal.json"
    path.write_text(
        json.dumps(
            dict(
                schema=1,
                prompt_ids=runner.requests["a"].prompt_token_ids,
                max_matches=1,
                output_directory=str(tmp_path / "trace"),
            )
        )
    )
    journal = ScoreJournal(path)
    runner._slimserve_score_journal = journal
    hidden = torch.randn(639, 17).bfloat16()
    method(runner, hidden, {"a": 640})
    journal.close()
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    records = [e for e in events if e["kind"] == "tensor"]
    assert [e["stage"] for e in records] == list(STAGES)
    expected = reference(
        hidden,
        torch.tensor(runner.requests["a"].prompt_token_ids[1:]),
        0,
        "raw_logprobs",
    )
    assert records[-1]["values"] == expected.logprobs.tolist()


def test_environment_defaults_to_original_and_enabled_is_model_scoped(
    runner_method, monkeypatch
):
    import os

    method, _ = runner_method
    runner, _, _ = make_runner(1025, 0, "raw_logprobs")
    runner.model_config.hf_config.model_type = "other"
    with pytest.raises(ValueError, match="GLM53 only"):
        method(runner, torch.zeros(1025, 17), {"a": 1026})
    # Missing flag leaves other models' existing prompt-score path intact.
    monkeypatch.delenv("SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS", raising=False)
    method.__globals__["os"] = os
    assert method(runner, torch.zeros(1025, 17), {"a": 1026})["a"].logprobs.shape == (
        1025,
        1,
    )
