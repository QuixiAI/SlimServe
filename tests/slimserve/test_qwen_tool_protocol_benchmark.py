# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen tool-protocol benchmark's accounting, driven by mocked SSE."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import benchmarks.benchmark_qwen_tool_protocol as bench

REPO_ROOT = Path(bench.__file__).resolve().parents[1]


def _args(tmp_path, turns=1):
    return SimpleNamespace(
        url="http://bench.test",
        model=bench.MODEL,
        concurrency=1,
        turns=turns,
        records=3,
        max_output_tokens=64,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=17,
        output=str(tmp_path / "out.json"),
    )


def _call(arguments):
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "record_summary",
        "arguments": arguments,
    }


def _sse(arguments, status="completed", reason=None):
    response = {
        "status": status,
        "output": [_call(arguments)],
        "usage": {"output_tokens": 10},
    }
    if reason:
        response["incomplete_details"] = {"reason": reason}
    events = [
        {"type": "response.created", "response": {"status": "in_progress"}},
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": arguments,
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "arguments": arguments,
        },
        {"type": f"response.{status}", "response": response},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return body + "data: [DONE]\n\n"


def _run(monkeypatch, tmp_path, bodies, turns=1):
    """Run bench.main against a transport replaying ``bodies`` in order."""
    replies = iter(bodies)

    def handler(request):
        return httpx.Response(
            200,
            text=next(replies),
            headers={"content-type": "text/event-stream"},
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        bench.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    args = _args(tmp_path, turns)
    exit_code = 0
    try:
        asyncio.run(bench.main(args))
    except SystemExit as exc:
        exit_code = exc.code
    raw = json.loads(Path(args.output).read_text())
    return raw["summary"], raw["results"], exit_code


def test_clean_rounds_count_as_completed(monkeypatch, tmp_path):
    good = json.dumps({"batch_id": "BATCH-0", "observations": []})
    summary, _, code = _run(monkeypatch, tmp_path, [_sse(good)] * 2, turns=2)
    assert summary["rounds"] == 2
    assert summary["completed_rounds"] == 2
    assert summary["protocol_errors"] == 0
    assert code == 0


def test_protocol_error_round_is_not_completed(monkeypatch, tmp_path):
    corrupted = json.dumps({"batch_id": "BATCH-9"})
    summary, results, code = _run(monkeypatch, tmp_path, [_sse(corrupted)])
    assert results[0]["rounds"][0]["errors"] == ["batch_id missing or corrupted"]
    assert summary["rounds"] == 1
    assert summary["protocol_errors"] == 1
    assert summary["completed_rounds"] == 0
    assert code == 1


def test_truncated_round_is_only_truncated(monkeypatch, tmp_path):
    # max_output_tokens cut the call mid-arguments: the partial JSON is not a
    # protocol fault and must not be validated.
    partial = '{"batch_id": "BATCH-0", "observa'
    summary, results, code = _run(
        monkeypatch,
        tmp_path,
        [_sse(partial, status="incomplete", reason="max_output_tokens")],
    )
    round_ = results[0]["rounds"][0]
    assert round_["incomplete_max_output_tokens"] is True
    assert round_["errors"] == []
    assert summary["incomplete_max_output_tokens"] == 1
    assert summary["protocol_errors"] == 0
    assert summary["completed_rounds"] == 0
    assert code == 1


def test_default_output_is_repo_rooted(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = bench.default_output()
    assert path.is_absolute()
    assert path.name == "qwen_tool_protocol.json"
    results = REPO_ROOT / "perf" / "results"
    assert path.parent.parent.parent == results
    assert len(path.parent.parent.name) == len("YYYY-MM-DD")


def test_default_output_run_ids_are_unique():
    paths = {bench.default_output() for _ in range(8)}
    assert len(paths) == 8


if __name__ == "__main__":
    pytest.main([__file__])
