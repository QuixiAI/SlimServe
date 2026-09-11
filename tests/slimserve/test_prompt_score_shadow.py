# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json

import pytest
import torch

from slimserve import prompt_score_shadow as shadow
from tests.slimserve import test_prompt_score_runner as runner_tests
from tests.slimserve.test_prompt_score_chunks import CPUSampler, reference
from tests.slimserve.test_prompt_score_runner import make_runner


@pytest.fixture
def runner_method(monkeypatch):
    return runner_tests.runner_method.__wrapped__(monkeypatch)


@pytest.fixture
def observer(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "VOCAB", 17)
    monkeypatch.setattr(shadow, "device_check", lambda _: None)
    value = shadow.PromptScoreShadow(tmp_path, 0)
    yield value
    value.stream.close()


@pytest.mark.parametrize("rows", [639, 1024, 1025, 2051])
@pytest.mark.parametrize("count", [0, 5])
def test_real_runner_shadow_exact_outputs_and_branch_coverage(
    observer, runner_method, rows, count
):
    method, _ = runner_method
    runner, projections, syncs = make_runner(rows, count, "raw_logprobs")
    runner._slimserve_prompt_score_shadow = observer
    logits = torch.randn(
        rows, 17, generator=torch.Generator().manual_seed(5310)
    ).bfloat16()
    before = logits.clone()
    ids = runner.requests["a"].prompt_token_ids
    expected = reference(logits, torch.tensor(ids[1:]), count, "raw_logprobs")
    actual = method(runner, logits, {"a": rows + 1})["a"]
    assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3], strict=True))
    assert torch.equal(before, logits) and len(projections) == 1 and syncs == [True]
    events = [json.loads(line) for line in observer.path.read_text().splitlines()]
    assert [e["kind"] for e in events] == ["header", "begin", "complete"]
    assert events[-1]["paired"] is (rows > 1024)
    assert bool(events[-1]["input_sha256"]) is (rows > 1024)
    assert events[-1]["token_ids"] == ids[1:]
    assert events[-1]["logprobs"] == actual.logprobs[:, 0].tolist()


@pytest.mark.parametrize("failure", ["scores", "ranks", "ids", "mutation", "nan"])
def test_shadow_fails_closed_and_retains_failure(observer, monkeypatch, failure):
    from vllm.v1.sample import prompt_logprobs

    real = prompt_logprobs.gather_prompt_logprobs

    def broken(logits, *args, **kwargs):
        output = real(logits, *args, **kwargs)
        if failure == "mutation":
            logits[0, 0] += 1
        elif failure == "nan":
            output.logprobs[0, 0] = float("nan")
        else:
            index = {"ids": 0, "scores": 1, "ranks": 2}[failure]
            output[index].view(-1)[0] += 1
        return output

    monkeypatch.setattr(prompt_logprobs, "gather_prompt_logprobs", broken)
    logits, targets = (
        torch.zeros(1025, 17).bfloat16(),
        torch.zeros(1025, dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="mutated|bits differ|nonfinite"):
        observer.gather(
            logits,
            targets,
            0,
            "raw_logprobs",
            sampler=CPUSampler,
            prompt_ids=[0] * 1026,
            start_idx=0,
            request_id="test",
        )
    events = [json.loads(line) for line in observer.path.read_text().splitlines()]
    assert [e["kind"] for e in events] == ["header", "begin", "failed"]


def test_hash_is_bounded_logical_bytes_and_distinguishes_signed_zero(monkeypatch):
    monkeypatch.setattr(shadow, "COPY_BYTES", 64)
    tensor = torch.arange(112, dtype=torch.float32).reshape(8, 14)[:, ::2]
    assert not tensor.is_contiguous()
    expected = hashlib.sha256(
        memoryview(tensor.contiguous().view(torch.uint8).numpy())
    ).hexdigest()
    assert shadow.tensor_digest(tensor) == expected
    assert shadow.tensor_digest(torch.tensor([0.0])) != shadow.tensor_digest(
        torch.tensor([-0.0])
    )
    with pytest.raises(ValueError, match="staging bound"):
        shadow.tensor_digest(torch.zeros(1, 17))


def test_disabled_is_inert_and_environment_is_scoped(tmp_path, monkeypatch):
    monkeypatch.delenv(shadow.FLAG, raising=False)
    assert shadow.PromptScoreShadow.from_env(object()) is None
    shadow.validate_plan(object())
    monkeypatch.setenv(shadow.FLAG, str(tmp_path))
    monkeypatch.setenv("SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS", "1")
    with pytest.raises(ValueError, match="under perf/results"):
        shadow.validate_environment()
    monkeypatch.setattr(shadow, "ROOT", tmp_path)
    monkeypatch.setenv(shadow.FLAG, str(tmp_path / "perf/results/shadow"))
    shadow.validate_environment()
    monkeypatch.setenv("SLIMSERVE_GLM53_NATIVE_ORDER", "1")
    with pytest.raises(ValueError, match="other model diagnostics"):
        shadow.validate_environment()


def test_real_registry_admits_only_fixed_recipe(tmp_path, monkeypatch):
    from slimserve.registry import resolve

    monkeypatch.setattr(shadow, "ROOT", tmp_path)
    monkeypatch.setenv(shadow.FLAG, str(tmp_path / "perf/results/shadow"))
    monkeypatch.setenv("SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS", "1")
    plan = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    shadow.validate_plan(plan)
    wrong = copy.deepcopy(plan)
    wrong.engine["compilation_config"]["inductor_compile_config"] = {
        "deterministic": True
    }
    with pytest.raises(ValueError, match="recipe/compiler policy"):
        shadow.validate_plan(wrong)


def test_partial_request_intervals_and_targets(observer, runner_method):
    method, _ = runner_method
    runner, projections, _ = make_runner(2051, 0, "raw_logprobs")
    runner._slimserve_prompt_score_shadow = observer
    logits = torch.zeros(2051, 17).bfloat16()
    assert method(runner, logits[:1025], {"a": 1025}) == {}
    runner.requests["a"].num_computed_tokens = 1025
    assert method(runner, logits[1025:], {"a": 1026}) == {}
    runner.requests["a"].num_computed_tokens = 2051
    assert method(runner, logits[:1], {"a": 1})["a"].logprobs.shape == (2051, 1)
    events = [json.loads(line) for line in observer.path.read_text().splitlines()]
    assert [e["start_idx"] for e in events if e["kind"] == "begin"] == [0, 1025]
    assert len(projections) == 2
