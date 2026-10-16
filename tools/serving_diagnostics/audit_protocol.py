#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit proxy captures locally; print aggregate counters, never payload content.

--since filters request start timestamps (Unix seconds or ISO-8601; naive ISO
values mean UTC). Elapsed totals sum request durations, not benchmark wall time.
Token totals use reported usage only; missing usage is counted, never estimated.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from capture_paths import DEFAULT_LOG_PATH

COUNTERS = (
    "log_records",
    "invalid_log_lines",
    "ignored_non_generation_records",
    "request_starts",
    "finished_requests",
    "pending_or_missing_finish_records",
    "http_errors",
    "proxy_errors",
    "stream_responses",
    "capture_truncated",
    "capture_truncated_without_terminal",
    "missing_terminal_without_capture_truncation",
    "invalid_sse_json_events",
    "terminal_completed",
    "terminal_incomplete",
    "terminal_failed",
    "terminal_other",
    "terminal_event_status_mismatch",
    "multiple_terminal_events",
    "stream_error_events",
    "required_completed_without_calls",
    "argument_done_events",
    "invalid_argument_done_json",
    "function_item_done_events",
    "invalid_function_item_done_json",
    "completed_invalid_function_item_done_json",
    "final_function_calls",
    "invalid_final_argument_json",
    "completed_invalid_final_argument_json",
    "argument_delta_done_mismatches",
    "argument_done_without_observed_delta",
    "usage_records",
    "missing_usage_records",
    "input_tokens",
    "output_tokens",
    "completed_usage_records",
    "completed_input_tokens",
    "completed_output_tokens",
    "elapsed_ms",
    "completed_elapsed_ms",
)
TERMINALS = {"response.completed", "response.incomplete", "response.failed"}


def timestamp(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (
            dt.replace(tzinfo=timezone.utc).timestamp()
            if dt.tzinfo is None
            else dt.timestamp()
        )


def reject_constant(_value):
    raise ValueError("Non-JSON numeric constant")


def valid_json(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        json.loads(value, parse_constant=reject_constant)
        return True
    except (ValueError, RecursionError):
        return False


def sse_events(body: str, counts: Counter):
    data = []
    for line in (body.replace("\r\n", "\n") + "\n\n").splitlines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            payload = "\n".join(data)
            data = []
            if payload == "[DONE]":
                yield {"type": "chat.done"}
                continue
            try:
                event = json.loads(payload)
            except (ValueError, RecursionError):
                counts["invalid_sse_json_events"] += 1
                continue
            if isinstance(event, dict):
                yield event


def audit_record(record: dict, counts: Counter) -> None:
    counts["finished_requests"] += 1
    status = record.get("status", 0)
    if isinstance(status, int) and status >= 400:
        counts["http_errors"] += 1
    counts["proxy_errors"] += bool(record.get("proxy_error"))
    elapsed = record.get("elapsed_ms", 0)
    if not isinstance(elapsed, (int, float)):
        elapsed = 0
    counts["elapsed_ms"] += elapsed
    request = record.get("request_body")
    request = request if isinstance(request, dict) else {}
    body = record.get("response_body")
    headers = record.get("response_headers", {})
    content_type = next(
        (v for k, v in headers.items() if k.lower() == "content-type"), ""
    )
    stream = isinstance(body, str) and (
        "text/event-stream" in content_type or "data:" in body
    )
    truncated = bool(record.get("response_capture_truncated"))
    counts["capture_truncated"] += truncated
    final = body if isinstance(body, dict) else None
    terminal_count = 0
    usage = None
    if stream:
        counts["stream_responses"] += 1
        deltas = {}
        index_ids = {}
        for event in sse_events(body, counts):
            kind = event.get("type", "")
            if kind in TERMINALS:
                terminal_count += 1
                final = event.get("response", {})
                final = final if isinstance(final, dict) else {}
                if final.get("status") != kind.split(".", 1)[1]:
                    counts["terminal_event_status_mismatch"] += 1
            elif kind == "chat.done":
                terminal_count += 1
                final = {"status": "completed"}
            elif kind in ("error", "response.error") or "error" in event:
                counts["stream_error_events"] += 1
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            item = event.get("item", {})
            if (
                kind == "response.output_item.added"
                and isinstance(item, dict)
                and item.get("id") is not None
            ):
                index_ids[event.get("output_index")] = item["id"]
            key = (
                event.get("item_id")
                or index_ids.get(event.get("output_index"))
                or event.get("output_index")
            )
            if kind == "response.function_call_arguments.delta":
                delta = event.get("delta", "")
                if isinstance(delta, str):
                    deltas[key] = deltas.get(key, "") + delta
            elif kind == "response.function_call_arguments.done":
                counts["argument_done_events"] += 1
                arguments = event.get("arguments")
                counts["invalid_argument_done_json"] += not valid_json(arguments)
                if key not in deltas:
                    counts["argument_done_without_observed_delta"] += 1
                elif deltas[key] != arguments:
                    counts["argument_delta_done_mismatches"] += 1
            elif (
                kind == "response.output_item.done"
                and isinstance(item, dict)
                and item.get("type") == "function_call"
            ):
                counts["function_item_done_events"] += 1
                invalid = not valid_json(item.get("arguments"))
                counts["invalid_function_item_done_json"] += invalid
                counts["completed_invalid_function_item_done_json"] += (
                    invalid and item.get("status") == "completed"
                )
        counts["multiple_terminal_events"] += terminal_count > 1
        if not terminal_count:
            counts[
                "capture_truncated_without_terminal"
                if truncated
                else "missing_terminal_without_capture_truncation"
            ] += 1
    elif isinstance(body, str):
        with suppress(ValueError, RecursionError):
            final = json.loads(body)
    final = final if isinstance(final, dict) else {}
    final_status = final.get("status")
    if final_status is None and isinstance(final.get("choices"), list):
        final_status = "completed"
    if final_status:
        counts[
            f"terminal_{final_status}"
            if final_status in ("completed", "incomplete", "failed")
            else "terminal_other"
        ] += 1
    output = final.get("output", [])
    output = output if isinstance(output, list) else []
    calls = [
        item
        for item in output
        if isinstance(item, dict) and str(item.get("type", "")).endswith("_call")
    ]
    # Chat-completion tools use a separate envelope.
    for choice in final.get("choices", []):
        if isinstance(choice, dict):
            calls.extend(choice.get("message", {}).get("tool_calls", []) or [])
    if (
        request.get("tool_choice") == "required"
        and final_status == "completed"
        and not calls
        # Chat SSE does not carry a final assembled choices list; its required
        # result cannot be inferred from this Responses-specific final envelope.
        and record.get("path") == "/v1/responses"
    ):
        counts["required_completed_without_calls"] += 1
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            counts["final_function_calls"] += 1
            invalid = not valid_json(item.get("arguments"))
            counts["invalid_final_argument_json"] += invalid
            counts["completed_invalid_final_argument_json"] += (
                invalid and item.get("status") == "completed"
            )
    usage = final.get("usage") or usage
    if isinstance(usage, dict):
        inputs = usage.get("input_tokens", usage.get("prompt_tokens"))
        outputs = usage.get("output_tokens", usage.get("completion_tokens"))
    else:
        inputs = outputs = None
    if type(inputs) is int and type(outputs) is int:
        counts["usage_records"] += 1
        counts["input_tokens"] += inputs
        counts["output_tokens"] += outputs
        if final_status == "completed":
            counts["completed_usage_records"] += 1
            counts["completed_input_tokens"] += inputs
            counts["completed_output_tokens"] += outputs
    else:
        counts["missing_usage_records"] += 1
    if final_status == "completed":
        counts["completed_elapsed_ms"] += elapsed


def audit(
    paths: list[Path], since: float = float("-inf"), until: float = float("inf")
) -> dict:
    counts = Counter({key: 0 for key in COUNTERS})
    starts = set()
    finishes = set()
    for path in paths:
        with path.open() as source:
            for line in source:
                try:
                    record = json.loads(line)
                except (ValueError, RecursionError):
                    counts["invalid_log_lines"] += 1
                    continue
                if (
                    not isinstance(record, dict)
                    or not since <= record.get("timestamp", 0) < until
                ):
                    continue
                counts["log_records"] += 1
                if record.get("path") not in ("/v1/responses", "/v1/chat/completions"):
                    counts["ignored_non_generation_records"] += 1
                    continue
                identity = (record.get("request_id"), record.get("timestamp"))
                if record.get("event") == "request_start":
                    counts["request_starts"] += 1
                    starts.add(identity)
                elif "status" in record:
                    finishes.add(identity)
                    audit_record(record, counts)
    counts["pending_or_missing_finish_records"] = len(starts - finishes)
    counts["elapsed_ms"] = round(counts["elapsed_ms"], 3)
    counts["completed_elapsed_ms"] = round(counts["completed_elapsed_ms"], 3)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        default=[DEFAULT_LOG_PATH],
    )
    parser.add_argument(
        "--since",
        type=timestamp,
        default=float("-inf"),
        help="Request start timestamp: Unix seconds or ISO-8601 (UTC by default)",
    )
    parser.add_argument(
        "--until",
        type=timestamp,
        default=float("inf"),
        help="Exclusive request start timestamp cutoff (same format as --since)",
    )
    args = parser.parse_args()
    print(
        json.dumps(audit(args.logs, args.since, args.until), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
