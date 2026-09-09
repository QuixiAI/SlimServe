# SPDX-License-Identifier: Apache-2.0
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

from slimserve.model_journal import (
    ACTIVE,
    OP_NAMES,
    ModelJournal,
    enabled,
    install_model_journal,
    instrument_mhc,
)
from slimserve.score_journal import ScoreJournal


@pytest.fixture
def journal(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "prompt_ids": list(range(640)),
                "max_matches": 3,
                "output_directory": str(tmp_path / "journals"),
            }
        )
    )
    score = ScoreJournal(path)
    model = ModelJournal(score)
    yield model
    model.close()
    score.close()


def test_disabled_is_original_callable_and_runner(monkeypatch):
    monkeypatch.delenv("SLIMSERVE_GLM53_MODEL_JOURNAL", raising=False)
    function = lambda: None
    assert instrument_mhc(function) is function
    runner = SimpleNamespace(_model_forward=function)
    install_model_journal(runner)
    assert runner._model_forward is function
    assert vars(runner) == {"_model_forward": function}


def test_invalid_flag_and_operation_rejected(monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    with pytest.raises(ValueError, match="unknown"):
        instrument_mhc(lambda: None)


def test_enabled_wrapper_preserves_signature_outputs_and_empty_context(monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")

    def glm5_mhc_pre(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return x

    wrapped = instrument_mhc(glm5_mhc_pre)
    assert inspect.signature(wrapped) == inspect.signature(glm5_mhc_pre)
    tensor = torch.ones(2)
    assert wrapped(tensor) is tensor
    calls = []
    observer = SimpleNamespace(
        before_op=lambda name, args: calls.append((name, args)),
        after_op=lambda name, value: calls.append((name, value)),
    )
    token = ACTIVE.set(observer)
    try:
        assert wrapped(tensor, scale=2.0) is tensor
    finally:
        ACTIVE.reset(token)
    assert calls[0][1] == {"x": tensor, "scale": 2.0}
    assert calls[1][1] is tensor
    assert ACTIVE.get() is None


def complete_ops(journal):
    tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    for site in range(91):
        name = OP_NAMES[0 if site == 0 else 1 if site < 90 else 2]
        journal.before_op(name, {"x": tensor, "eps": 1e-5})
        result = tensor if site == 90 else (tensor,) * (3 if site == 0 else 4)
        journal.after_op(name, result)


def test_all_operations_and_repeats_are_bounded(journal):
    for match in range(3):
        journal.begin(f"request-{match}")
        complete_ops(journal)
        journal.finish()
    with pytest.raises(ValueError, match="extra"):
        journal.begin("fourth")
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert rows[0]["expected_operations"] == 91
    assert len([r for r in rows if r["kind"] == "complete"]) == 3
    assert len([r for r in rows if r["kind"] == "operation"]) == 3 * 91
    assert {r["parameters"]["eps"] for r in rows if r["kind"] == "operation"} == {1e-5}
    assert len({r["sha256"] for r in rows if r["kind"] == "tensor"}) == 1


def test_inactive_overlap_incomplete_and_wrong_operation_rejected(journal):
    with pytest.raises(ValueError, match="inactive"):
        journal.record("bad", torch.ones(1))
    with pytest.raises(ValueError, match="incomplete"):
        journal.finish()
    journal.begin("one")
    with pytest.raises(ValueError, match="overlapping"):
        journal.begin("two")
    with pytest.raises(ValueError, match="order"):
        journal.before_op(OP_NAMES[1], {})
    journal.before_op(OP_NAMES[0], {})
    with pytest.raises(ValueError, match="order"):
        journal.before_op(OP_NAMES[0], {})
    with pytest.raises(ValueError, match="completion"):
        journal.after_op(OP_NAMES[1], ())
    with pytest.raises(ValueError, match="arity"):
        journal.after_op(OP_NAMES[0], (torch.ones(1),))
    with pytest.raises(ValueError, match="incomplete"):
        journal.finish()


def test_extra_operation_refused(journal):
    journal.begin("one")
    complete_ops(journal)
    with pytest.raises(ValueError, match="order"):
        journal.before_op(OP_NAMES[2], {})


@pytest.mark.parametrize("shape", [(0,), (2**29,)])
def test_byte_bound_checked_before_copy(journal, shape):
    journal.begin("one")
    with pytest.raises(ValueError, match="byte bound"):
        journal.record("bad", torch.empty(shape, device="meta"))


def test_record_bound_checked_before_copy(journal):
    journal.begin("one")
    journal.records = 1024
    with pytest.raises(ValueError, match="record bound"):
        journal.record("bad", torch.empty(1, device="meta"))


@pytest.mark.parametrize("change", ["model", "tp", "pp", "spec", "score_config"])
def test_install_scope_is_validated(monkeypatch, change):
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    monkeypatch.delenv("SLIMSERVE_GLM53_SCORE_JOURNAL", raising=False)
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="glm5_next"),
            hf_text_config=SimpleNamespace(hidden_size=4096, num_hidden_layers=45),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, pipeline_parallel_size=1
        ),
        speculative_config=None,
    )
    if change == "model":
        runner.model_config.hf_text_config.hidden_size = 2048
    elif change == "tp":
        runner.parallel_config.tensor_parallel_size = 2
    elif change == "pp":
        runner.parallel_config.pipeline_parallel_size = 2
    elif change == "spec":
        runner.speculative_config = object()
    with pytest.raises(ValueError):
        install_model_journal(runner)
