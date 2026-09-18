# SPDX-License-Identifier: Apache-2.0
"""Replay the valid prefixes of captured malformed Responses requests.

The 4xx diagnostic log contains the complete request body.  For each selected
record, this script finds the first function call whose ``arguments`` are not
valid JSON, removes that item and everything after it, then submits the exact
preceding conversation as a streaming Responses request.  Raw SSE and a
content-free consistency summary are written with mode 0600.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _write_private(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(data)


def _is_valid_json(arguments: Any) -> bool:
    if not isinstance(arguments, str):
        return False
    try:
        json.loads(arguments)
    except (TypeError, ValueError):
        return False
    return True


def _has_identical_halves(value: str) -> bool:
    midpoint = len(value) // 2
    return bool(value) and len(value) % 2 == 0 and value[:midpoint] == value[midpoint:]


def _valid_prefix(request: dict[str, Any]) -> tuple[dict[str, Any], int]:
    items = request.get("input")
    if not isinstance(items, list):
        raise ValueError("request input is not an item list")

    cut = None
    for index, item in enumerate(items):
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and not _is_valid_json(item.get("arguments"))
        ):
            cut = index
            break
    if cut is None:
        raise ValueError("request has no malformed function_call arguments")

    replay = dict(request)
    replay["input"] = items[:cut]
    replay["stream"] = True
    return replay, cut


def _parse_sse(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


def _argument_fingerprint(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    decoded: Any = None
    valid = False
    try:
        decoded = json.loads(value)
        valid = True
    except (TypeError, ValueError):
        pass
    return {
        "bytes": len(value.encode()),
        "sha256": hashlib.sha256(value.encode()).hexdigest(),
        "valid_json": valid,
        "json_object": valid and isinstance(decoded, dict),
        "identical_halves": _has_identical_halves(value),
    }


def _analyze(events: list[dict[str, Any]]) -> dict[str, Any]:
    calls: dict[int, dict[str, Any]] = {}
    completed_arguments: list[str] = []
    for event in events:
        event_type = event.get("type")
        output_index = event.get("output_index")
        if isinstance(output_index, int):
            call = calls.setdefault(
                output_index,
                {"name": None, "deltas": [], "done": None, "item_done": None},
            )
            if event_type == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    call["name"] = item.get("name")
            elif event_type == "response.function_call_arguments.delta":
                call["deltas"].append(event.get("delta", ""))
            elif event_type == "response.function_call_arguments.done":
                call["done"] = event.get("arguments")
                call["name"] = event.get("name") or call["name"]
            elif event_type == "response.output_item.done":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    call["item_done"] = item.get("arguments")
                    call["name"] = item.get("name") or call["name"]
        if event_type == "response.completed":
            for item in event.get("response", {}).get("output", []):
                if item.get("type") == "function_call":
                    completed_arguments.append(item.get("arguments", ""))

    summaries = []
    for output_index, call in sorted(calls.items()):
        joined = "".join(call["deltas"])
        done = call["done"]
        item_done = call["item_done"]
        summaries.append(
            {
                "output_index": output_index,
                "name": call["name"],
                "delta_count": len(call["deltas"]),
                "delta_join": _argument_fingerprint(joined),
                "done": _argument_fingerprint(done),
                "output_item_done": _argument_fingerprint(item_done),
                "done_matches_delta_join": done == joined,
                "output_item_done_matches_delta_join": item_done == joined,
            }
        )

    completed = [_argument_fingerprint(value) for value in completed_arguments]
    return {
        "event_count": len(events),
        "event_types": [event.get("type") for event in events],
        "streamed_calls": summaries,
        "completed_calls": completed,
        "completed_count": len(completed_arguments),
        "completed_matches_stream_order": completed_arguments
        == ["".join(call["deltas"]) for _, call in sorted(calls.items())],
    }


def _replay_one(
    record_number: int,
    request: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    output_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    replay, cut = _valid_prefix(request)
    stem = f"record-{record_number:02d}"
    _write_private(
        output_dir / f"{stem}-request.json",
        json.dumps(replay, ensure_ascii=False, indent=2) + "\n",
    )
    http_request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/responses",
        data=json.dumps(replay, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Request-Id": f"replay-bad-prefix-{record_number:02d}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            raw = response.read().decode(errors="replace")
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        status = error.code

    suffix = "sse" if status == 200 else "error.txt"
    _write_private(output_dir / f"{stem}.{suffix}", raw)
    result: dict[str, Any] = {
        "record": record_number,
        "status": status,
        "input_items_before": len(request.get("input", [])),
        "input_items_replayed": cut,
    }
    if status == 200:
        result.update(_analyze(_parse_sse(raw)))
    else:
        result["response_bytes"] = len(raw.encode())
        result["response_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--first-record", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api-key or OPENAI_API_KEY")

    selected: list[tuple[int, dict[str, Any]]] = []
    for record_number, line in enumerate(args.diagnostics.read_text().splitlines(), 1):
        if record_number < args.first_record:
            continue
        record = json.loads(line)
        request_body = record.get("request_body")
        if not isinstance(request_body, str):
            continue
        request = json.loads(request_body)
        try:
            _valid_prefix(request)
        except ValueError:
            continue
        selected.append((record_number, request))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _replay_one,
                record_number,
                request,
                base_url=args.base_url,
                api_key=args.api_key,
                output_dir=args.output_dir,
                timeout=args.timeout,
            )
            for record_number, request in selected
        ]
        completed_futures = concurrent.futures.as_completed(futures)
        results = [future.result() for future in completed_futures]

    results.sort(key=lambda item: item["record"])
    summary = {"replays": results}
    _write_private(
        args.output_dir / "summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
