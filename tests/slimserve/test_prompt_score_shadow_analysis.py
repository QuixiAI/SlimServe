# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from benchmarks import analyze_glm53_prompt_score_shadow as analysis
from benchmarks import run_glm53_prompt_score_shadow as runner
from tests.slimserve import test_quality_pair_analysis as quality_tests


@pytest.fixture(scope="module")
def quality():
    doc = quality_tests.build_docs()[0]
    for row in doc["needles"]:
        row["prefix_ids"][0] = 1 if row["position"] == 0.25 else 2
    requests = [(r["prompt_ids"], r["response"]) for r in doc["text"]]
    requests += [
        (r["prefix_ids"] + c["suffix_ids"], c["response"])
        for r in doc["needles"]
        for c in r["candidates"]
    ]
    for ids, response in requests:
        response["usage"]["prompt_tokens_details"] = {"cached_tokens": 0}
        scores = response["choices"][0]["prompt_logprobs"]
        for i, token in enumerate(ids[1:], 1):
            if scores[i] is None:
                scores[i] = {str(token): {"logprob": -2.0}}
            scores[i][str(token)]["rank"] = 1
    return doc


@pytest.fixture(scope="module")
def evidence(quality):
    source_names = (
        "slimserve/prompt_score_shadow.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/sample/prompt_logprobs.py",
        "vllm/v1/sample/sampler.py",
        "vllm/v1/sample/ops/logprobs.py",
    )
    sources = dict.fromkeys(source_names, "source-digest")
    events = [
        dict(
            kind="header",
            schema=1,
            rank=0,
            diagnostic_only=True,
            vocab=154880,
            max_rows=8192,
            copy_bytes=32 * 1024**2,
            sources=sources,
        )
    ]
    call = 0
    for key, request in analysis.request_inventory(quality).items():
        for start in range(0, len(request["ids"]) - 1, 7616):
            call += 1
            tokens = request["ids"][start + 1 : start + 7617]
            scores, ranks = (
                request["scores"][start : start + 7616],
                request["ranks"][start : start + 7616],
            )
            rows = len(tokens)
            events.append(
                dict(
                    kind="begin",
                    call=call,
                    request_id=key,
                    prompt_sha256=key,
                    prompt_tokens=len(request["ids"]),
                    start_idx=start,
                    rows=rows,
                )
            )
            outputs = [
                dict(
                    shape=shape,
                    dtype=dtype,
                    reference_sha256=analysis.digest(values, code),
                    chunked_sha256=analysis.digest(values, code),
                )
                for values, code, dtype, shape in zip(
                    (tokens, scores, ranks),
                    ("i", "f", "q"),
                    ("torch.int32", "torch.float32", "torch.int64"),
                    ([rows, 1], [rows, 1], [rows]),
                    strict=True,
                )
            ]
            events.append(
                dict(
                    kind="complete",
                    call=call,
                    paired=rows > 1024,
                    count=0,
                    mode="raw_logprobs",
                    input_sha256=["a" * 64, analysis.digest(tokens, "q")]
                    if rows > 1024
                    else None,
                    outputs=outputs,
                    token_ids=tokens,
                    logprobs=scores,
                    ranks=ranks,
                )
            )
    return events, sources


@pytest.mark.parametrize(
    "mutation",
    [None, "rank", "hash", "coverage", "interval", "api", "truncated", "missing"],
)
def test_offline_join_covers_all_requests_and_rejects_bad_evidence(
    tmp_path, quality, evidence, mutation
):
    events, sources = evidence
    text = "\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n"
    for rank in range(4):
        path = tmp_path / f"shadow-rank{rank}-pid100.jsonl"
        if mutation == "missing" and rank == 3:
            continue
        header = dict(events[0], rank=rank)
        rest = text.split("\n", 1)[1]
        if rank == 0 and mutation:
            changed = json.loads(json.dumps(events))
            if mutation == "rank":
                header["rank"] = 1
            elif mutation == "hash":
                changed[2]["outputs"][0]["reference_sha256"] = "b" * 64
            elif mutation == "coverage":
                changed[2]["paired"] = True
            elif mutation == "interval":
                changed[1]["start_idx"] = 1
            elif mutation == "api":
                changed[2]["logprobs"][0] -= 0.5
            elif mutation == "truncated":
                changed.pop()
            rest = (
                "\n".join(json.dumps(e, separators=(",", ":")) for e in changed[1:])
                + "\n"
            )
        path.write_text(json.dumps(header) + "\n" + rest)
    if mutation:
        with pytest.raises(ValueError):
            analysis.audit(tmp_path, quality, sources)
    else:
        result = analysis.audit(tmp_path, quality, sources)
        assert result["exact_live_scorer_parity"]
        assert {r["rank"] for r in result["ranks"]} == set(range(4))
        assert all(
            r["requests"] == 56 and r["paired_calls"] > 0 for r in result["ranks"]
        )


@pytest.mark.parametrize("failure", [None, "serve", "audit", "interrupt"])
def test_controller_one_attempt_preserves_terminal_artifacts(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.rollout, "historical_evidence", lambda: [{}, {}])
    monkeypatch.setattr(runner.rollout, "freeze", lambda: {"gpu_config": "fixed"})
    monkeypatch.setattr(runner, "extra_sources", lambda: {"fixed": "source"})
    monkeypatch.setattr(runner.rollout, "check_freeze", lambda _: None)
    monkeypatch.setattr(runner.rollout, "gpu_processes", lambda: "")
    monkeypatch.setattr(runner.rollout, "gpu_config", lambda: "fixed")
    folder = tmp_path / "perf/results/shadow"
    monkeypatch.setattr("sys.argv", ["shadow", "--output", str(folder)])
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "systemctl":
            return SimpleNamespace(returncode=0)
        assert "MemoryMax=150G" in argv and "MemorySwapMax=0" in argv
        assert kwargs["env"][runner.FLAG] == str(folder / "shadow")
        assert kwargs["env"]["SLIMSERVE_GLM53_NATIVE_ORDER"] == "0"
        kwargs["stdout"].write("preserved attempt\n")
        if failure == "interrupt":
            raise RuntimeError("interrupted")
        return SimpleNamespace(returncode=int(failure == "serve"))

    def audit(*args):
        if failure == "audit":
            raise ValueError("mismatch")
        return dict(
            shadow={"exact_live_scorer_parity": True},
            historical_quality={"passed": False},
        )

    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(runner, "audit", audit)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            runner.main()
    else:
        runner.main()
    report = json.loads((folder / "result.json").read_text())
    assert report["status"] == ("failed" if failure else "complete")
    assert report["promotion"] is False and report["source_freeze_verified"]
    assert len([c for c in calls if c[0] == "systemd-run"]) == 1
    assert bool([c for c in calls if c[0] == "systemctl"]) is (failure == "interrupt")
    assert (folder / "serve.log").read_text() == "preserved attempt\n"
    with pytest.raises(ValueError, match="new result"):
        runner.main()
