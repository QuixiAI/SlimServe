# SPDX-License-Identifier: Apache-2.0
"""A time-capped long run is not proof of actual target-depth recall."""

import asyncio
import copy
import json
import sys
import time
from types import SimpleNamespace

import pytest

from benchmarks import benchmark_wildchat_deepcontext as bench


def successful_records(sid=0):
    return [
        dict(
            session=sid,
            prompt_tokens=1_000_000,
            completion_tokens=40,
            probe=True,
            target_probe=True,
            recall_ok=True,
        ),
        dict(session=sid, final=True, ended="target", ctx_tokens=1_000_040),
    ]


def test_every_session_needs_actual_target_depth_recall():
    records = successful_records() + successful_records(1)
    assert bench.qualify_target(records, 2, 1_000_000)["passed"]
    records[2]["prompt_tokens"] = 999_999
    records[2]["completion_tokens"] = 1000
    assert not bench.qualify_target(records, 2, 1_000_000)["passed"]


@pytest.mark.parametrize(
    "problem",
    [
        "wall_clock",
        "ceiling",
        "recall",
        "shallow_probe",
        "missing_usage",
        "zero_usage",
        "error",
        "missing_session",
        "duplicate_final",
        "unknown_session",
    ],
)
def test_incomplete_or_incorrect_evidence_fails(problem):
    records = copy.deepcopy(successful_records())
    if problem in ("wall_clock", "ceiling"):
        records[-1]["ended"] = problem
    elif problem == "recall":
        records[0]["recall_ok"] = False
    elif problem == "shallow_probe":
        records[0]["prompt_tokens"] = 530_222
    elif problem == "missing_usage":
        del records[0]["completion_tokens"]
    elif problem == "zero_usage":
        records[0]["completion_tokens"] = 0
    elif problem == "error":
        records.append(dict(session=0, error="APIError: engine failure"))
    elif problem == "missing_session":
        records.clear()
    elif problem == "duplicate_final":
        records.append(records[-1].copy())
    elif problem == "unknown_session":
        records.append(dict(session=1, final=True, ended="target"))
    assert not bench.qualify_target(records, 1, 1_000_000)["passed"]


def test_reasoning_completion_usage_cannot_fake_target_depth(monkeypatch):
    # First growth crosses the target only when reasoning completion tokens
    # are added. The first deep probe is too shallow, so growth must continue.
    counts = iter(
        [(10, 1), (950_000, 60_000), (951_000, 20), (1_010_000, 20), (1_010_100, 20)]
    )
    markers = []

    async def fake_turn(
        client,
        model,
        messages,
        records,
        sid,
        depth,
        args,
        max_tokens,
        probe_marker=None,
        target_probe=False,
    ):
        prompt, completion = next(counts)
        record = dict(
            session=sid, depth=depth, prompt_tokens=prompt, completion_tokens=completion
        )
        if probe_marker is not None:
            markers.append(probe_marker)
            record.update(probe=True, recall_ok=True, target_probe=target_probe)
        records.append(record)
        return probe_marker or "ok"

    monkeypatch.setattr(bench, "_turn", fake_turn)
    args = SimpleNamespace(
        seed=42,
        require_target=True,
        probe_every=6,
        paste_chars=10,
        reply_tokens=512,
        ctx_target=1_000_000,
    )
    records = []
    asyncio.run(
        bench.run_session(
            None,
            "model",
            0,
            ["question"],
            ["document"],
            records,
            time.time() + 10,
            args,
            asyncio.Semaphore(1),
        )
    )
    assert len(markers) == 2 and markers[0] == markers[1]
    assert records[-1]["ended"] == "target"
    assert bench.qualify_target(records, 1, 1_000_000)["passed"]


def test_strict_mode_does_not_squeeze_after_arbitrary_request_error(monkeypatch):
    calls = 0

    async def failing_turn(
        client, model, messages, records, sid, depth, args, max_tokens, **kwargs
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            records.append(dict(session=sid, prompt_tokens=20, completion_tokens=2))
            return "ok"
        records.append(dict(session=sid, error="APIError: engine failure"))
        return None

    monkeypatch.setattr(bench, "_turn", failing_turn)
    args = SimpleNamespace(
        seed=42,
        require_target=True,
        probe_every=6,
        paste_chars=10,
        reply_tokens=512,
        ctx_target=1_000_000,
    )
    records = []
    asyncio.run(
        bench.run_session(
            None,
            "model",
            0,
            ["question"],
            ["document"],
            records,
            time.time() + 10,
            args,
            asyncio.Semaphore(1),
        )
    )
    assert calls == 2
    assert records[-1]["ended"] == "error"
    assert not bench.qualify_target(records, 1, 1_000_000)["passed"]


@pytest.mark.parametrize(
    "usage", [None, SimpleNamespace(prompt_tokens=0, completion_tokens=20)]
)
def test_strict_turn_rejects_missing_usage_immediately(usage):
    async def events():
        yield SimpleNamespace(
            usage=usage,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="marker", reasoning_content=None)
                )
            ],
        )

    async def create(**kwargs):
        return events()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    records = []
    result = asyncio.run(
        bench._turn(
            client,
            "model",
            [],
            records,
            0,
            0,
            SimpleNamespace(require_target=True, turn_timeout=1),
            1024,
            probe_marker="marker",
            target_probe=True,
        )
    )
    assert result is None
    assert records[-1]["error"].startswith("InvalidUsage:")


@pytest.mark.parametrize("passed", [False, True])
def test_cli_persists_qualification_before_nonzero_exit(monkeypatch, tmp_path, passed):
    output = tmp_path / "qualification.json"

    async def list_models():
        return SimpleNamespace(data=[SimpleNamespace(id="model")])

    client = SimpleNamespace(models=SimpleNamespace(list=list_models))
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda **kwargs: client)
    )
    monkeypatch.setattr(bench, "load_pools", lambda *args: (["q"], ["document"]))

    async def session(client, model, sid, turns, pastes, records, stop_at, args, sem):
        probe, final = successful_records(sid)
        probe.update(ttft_s=0.1, e2e_s=1.0)
        if not passed:
            final["ended"] = "wall_clock"
        records.extend([probe, final])

    monkeypatch.setattr(bench, "run_session", session)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_wildchat_deepcontext.py",
            "--parquet",
            "unused",
            "--concurrency",
            "1",
            "--ctx-target",
            "1000000",
            "--require-target",
            "--out",
            str(output),
        ],
    )
    if passed:
        asyncio.run(bench.main())
    else:
        with pytest.raises(SystemExit) as exc:
            asyncio.run(bench.main())
        assert exc.value.code == 1
    saved = json.loads(output.read_text())
    assert len(saved["records"]) == 2
    assert saved["summary"]["target_qualification"]["passed"] is passed


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_ttft_includes_either_reasoning_field(monkeypatch, field):
    clock = [0.0]
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])

    async def events():
        clock[0] = 1.0
        yield SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, **{field: "thinking"})
                )
            ],
        )
        clock[0] = 3.0
        yield SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="marker"))],
        )
        clock[0] = 4.0
        yield SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20), choices=[]
        )

    async def create(**kwargs):
        return events()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    records = []
    assert (
        asyncio.run(
            bench._turn(
                client,
                "model",
                [],
                records,
                0,
                0,
                SimpleNamespace(require_target=True, turn_timeout=1),
                1024,
                probe_marker="marker",
                target_probe=True,
            )
        )
        == "marker"
    )
    assert records[0]["ttft_s"] == 1.0
    assert records[0]["e2e_s"] == 4.0
