#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only compact assistant/response diagnostics; never print payload text.

Argument SHA256 fingerprints link replayed calls without exposing their code or
values. JSON syntax checks do not enforce tool schemas or repair arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from audit_protocol import TERMINALS, reject_constant, sse_events, timestamp
from capture_paths import DEFAULT_LOG_PATH

KNOWN_TYPES = {
    "message",
    "function_call",
    "custom_tool_call",
    "reasoning",
    "output_text",
    "input_text",
    "text",
    "refusal",
    "reasoning_text",
    "summary_text",
    "function",
}
KNOWN_STATUSES = {"completed", "incomplete", "in_progress", "failed", "queued"}
KNOWN_EVENTS = TERMINALS | {
    "response.created",
    "response.in_progress",
    "response.queued",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.reasoning_part.added",
    "response.reasoning_part.done",
    "response.reasoning_text.delta",
    "response.reasoning_text.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.refusal.delta",
    "response.refusal.done",
    "error",
    "response.error",
    "chat.done",
}


def identifier(value):
    if not isinstance(value, str):
        return None
    if len(value) <= 128 and re.fullmatch(r"[A-Za-z0-9_.:/-]+", value):
        return value
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def fingerprint(value):
    if not isinstance(value, str):
        return {"string": False, "value_type": type(value).__name__}
    return {
        "string": True,
        "chars": len(value),
        "bytes": len(value.encode()),
        "sha256": hashlib.sha256(value.encode()).hexdigest(),
    }


def arguments_summary(value):
    result = fingerprint(value)
    result["valid_json"] = False
    if not isinstance(value, str):
        return result
    try:
        parsed = json.loads(value, parse_constant=reject_constant)
        result.update(valid_json=True, json_type=type(parsed).__name__)
    except json.JSONDecodeError as exc:
        result.update(
            error_type="JSONDecodeError",
            error_pos=exc.pos,
            error_line=exc.lineno,
            error_column=exc.colno,
        )
    except (ValueError, RecursionError) as exc:
        result["error_type"] = type(exc).__name__
    return result


def assistant_item(item, position):
    if not isinstance(item, dict):
        return None
    kind = item.get("type", "message")
    if item.get("role") != "assistant" and kind not in {
        "function_call",
        "custom_tool_call",
        "reasoning",
    }:
        return None
    result = {"position": position, "type": kind if kind in KNOWN_TYPES else "other"}
    for key in ("id", "call_id"):
        if key in item:
            result[key] = identifier(item[key])
    if item.get("status") in KNOWN_STATUSES:
        result["status"] = item["status"]
    if kind == "function_call":
        result["arguments"] = arguments_summary(item.get("arguments"))
    if kind == "custom_tool_call":
        result["input"] = fingerprint(item.get("input"))
    if "content" in item:
        content = item["content"]
        if isinstance(content, list):
            result["content_parts"] = [
                {
                    "position": index,
                    "type": part.get("type")
                    if part.get("type") in KNOWN_TYPES
                    else "other",
                    **fingerprint(part.get("text", part.get("refusal"))),
                }
                for index, part in enumerate(content)
                if isinstance(part, dict)
            ]
            result["content_part_count"] = len(content)
        else:
            result["content"] = fingerprint(content)
    if isinstance(item.get("tool_calls"), list):
        result["tool_calls"] = [
            {
                "position": index,
                "id": identifier(call.get("id")),
                "arguments": arguments_summary(
                    call.get("function", {}).get("arguments")
                ),
            }
            for index, call in enumerate(item["tool_calls"])
            if isinstance(call, dict) and isinstance(call.get("function"), dict)
        ]
    return result


def selected_items(items):
    if not isinstance(items, list):
        return []
    return [
        summary
        for index, item in enumerate(items)
        if (summary := assistant_item(item, index)) is not None
    ]


def compare_arguments(left, right):
    if not isinstance(left, str) or not isinstance(right, str):
        return {"equal": False, "comparable_strings": False}
    if left == right:
        return {"equal": True}
    first = next(
        (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )
    return {
        "equal": False,
        "first_difference_char": first,
        "left_chars": len(left),
        "right_chars": len(right),
    }


def response_error(error):
    if not isinstance(error, dict):
        return {"present": bool(error)}
    message = error.get("message")
    result = {"present": True, "message": fingerprint(message)}
    if isinstance(message, str):
        match = re.search(r"line (\d+) column (\d+) \(char (\d+)\)", message)
        if match:
            result.update(
                json_error_line=int(match[1]),
                json_error_column=int(match[2]),
                json_error_pos=int(match[3]),
            )
        result["list_index_out_of_range"] = "list index out of range" in message
    return result


def inspect_record(record):
    request = record.get("request_body")
    request = request if isinstance(request, dict) else {}
    result = {
        "request_id": identifier(record.get("request_id")),
        "timestamp": record.get("timestamp"),
        "http_status": record.get("status"),
        "elapsed_ms": record.get("elapsed_ms"),
        "capture_truncated": bool(record.get("response_capture_truncated")),
        "assistant_history": selected_items(
            request.get("input", request.get("messages"))
        ),
        "response": {},
    }
    response = result["response"]
    body = record.get("response_body")
    headers = record.get("response_headers", {})
    headers = headers if isinstance(headers, dict) else {}
    response["headers"] = {
        key.lower(): value
        if isinstance(value, str)
        and len(value) <= 100
        and re.fullmatch(r"[A-Za-z0-9/+;,= ._-]+", value)
        else "other"
        for key, value in headers.items()
        if key.lower() in {"content-type", "content-encoding"}
    }
    response["body_chars"] = len(body) if isinstance(body, str) else None
    # The proxy currently records only total request elapsed time, not chunk
    # arrival times. Never infer TTFT from an SSE sequence number or count.
    response["first_chunk_ms"] = record.get("first_chunk_ms")
    stream = isinstance(body, str) and (
        any(
            "text/event-stream" in str(value)
            for key, value in headers.items()
            if key.lower() == "content-type"
        )
        or body.lstrip().startswith(("event:", "data:"))
    )
    final = body if isinstance(body, dict) else {}
    if stream:
        counts = Counter()
        calls = {}
        index_ids = {}
        terminal_types = []
        event_counts = Counter()
        first_events = []
        for event in sse_events(body, counts):
            kind = event.get("type")
            safe_kind = (
                kind if isinstance(kind, str) and kind in KNOWN_EVENTS else "other"
            )
            event_counts[safe_kind] += 1
            if len(first_events) < 8:
                summary = {"type": safe_kind}
                if type(event.get("sequence_number")) is int:
                    summary["sequence_number"] = event["sequence_number"]
                event_response = event.get("response")
                if isinstance(event_response, dict) and type(
                    event_response.get("created_at")
                ) in (int, float):
                    summary["response_created_at"] = event_response["created_at"]
                first_events.append(summary)
            item = event.get("item")
            item = item if isinstance(item, dict) else {}
            if kind in {
                "response.output_item.added",
                "response.output_item.done",
            } and item.get("id"):
                index_ids[event.get("output_index")] = item["id"]
            key = (
                event.get("item_id")
                or item.get("id")
                or index_ids.get(event.get("output_index"))
            )
            if key is None:
                key = "index:" + str(event.get("output_index"))
            if kind in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }:
                call = calls.setdefault(key, {})
                if kind.endswith(".delta"):
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        call["delta"] = call.get("delta", "") + delta
                        call["delta_events"] = call.get("delta_events", 0) + 1
                else:
                    call["done"] = event.get("arguments")
            if (
                kind == "response.output_item.done"
                and item.get("type") == "function_call"
            ):
                calls.setdefault(key, {})["item_done"] = item.get("arguments")
            if kind in TERMINALS:
                terminal_types.append(kind)
                final = event.get("response", {})
            if kind in {"error", "response.error"} or "error" in event:
                response.setdefault("errors", []).append(
                    response_error(event.get("error", event))
                )
        if not isinstance(final, dict):
            final = {}
        for item in final.get("output", []):
            if isinstance(item, dict) and item.get("type") == "function_call":
                calls.setdefault(item.get("id"), {})["final"] = item.get("arguments")
        response["terminal_events"] = terminal_types
        response["first_events"] = first_events
        response["event_counts"] = dict(event_counts)
        finished_capture = record.get("event") != "request_start" and "status" in record
        response["unterminated_finished_stream"] = bool(
            finished_capture and not terminal_types and not event_counts["chat.done"]
        )
        response["missing_terminal_without_capture_truncation"] = bool(
            response["unterminated_finished_stream"] and not result["capture_truncated"]
        )
        response["invalid_sse_events"] = counts["invalid_sse_json_events"]
        response["stream_calls"] = []
        for key, values in calls.items():
            call = {
                "item_id": identifier(key),
                "delta_events": values.get("delta_events", 0),
            }
            for stage in ("delta", "done", "item_done", "final"):
                if stage in values:
                    call[stage] = arguments_summary(values[stage])
            for left, right in (
                ("delta", "done"),
                ("delta", "item_done"),
                ("delta", "final"),
                ("done", "final"),
                ("item_done", "final"),
            ):
                if left in values and right in values:
                    call[f"{left}_vs_{right}"] = compare_arguments(
                        values[left], values[right]
                    )
            response["stream_calls"].append(call)
    elif isinstance(body, str):
        response["body_json"] = arguments_summary(body)
        try:
            final = json.loads(body)
        except (ValueError, RecursionError):
            final = {}
    final = final if isinstance(final, dict) else {}
    response["id"] = identifier(final.get("id"))
    response["status"] = (
        final.get("status") if final.get("status") in KNOWN_STATUSES else None
    )
    response["output"] = selected_items(final.get("output"))
    if isinstance(final.get("choices"), list):
        response["chat_messages"] = selected_items(
            [
                choice.get("message")
                for choice in final["choices"]
                if isinstance(choice, dict)
            ]
        )
    if "error" in final:
        response["error"] = response_error(final["error"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        default=[DEFAULT_LOG_PATH],
    )
    parser.add_argument("--since", type=timestamp, default=float("-inf"))
    parser.add_argument("--until", type=timestamp, default=float("inf"))
    parser.add_argument("--request-id")
    parser.add_argument(
        "--errors-only", action="store_true", help="Only HTTP errors (status >= 400)"
    )
    parser.add_argument(
        "--unterminated-only",
        action="store_true",
        help="Only finished SSE captures lacking a terminal (includes HTTP 200)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum finished capture records to print",
    )
    args = parser.parse_args()
    printed = 0
    invalid_lines = 0
    for path in args.paths:
        if printed >= args.limit:
            break
        with path.open() as source:
            for line in source:
                if printed >= args.limit:
                    break
                try:
                    record = json.loads(line)
                except (ValueError, RecursionError):
                    invalid_lines += 1
                    continue
                if (
                    not isinstance(record, dict)
                    or record.get("event") == "request_start"
                ):
                    continue
                if record.get("path") not in {"/v1/responses", "/v1/chat/completions"}:
                    continue
                if not args.since <= record.get("timestamp", 0) < args.until:
                    continue
                if args.request_id and record.get("request_id") != args.request_id:
                    continue
                if args.errors_only and record.get("status", 0) < 400:
                    continue
                inspected = inspect_record(record)
                if args.unterminated_only and not inspected["response"].get(
                    "unterminated_finished_stream"
                ):
                    continue
                print(json.dumps(inspected, separators=(",", ":")))
                printed += 1
    print(
        json.dumps({"records_displayed": printed, "invalid_log_lines": invalid_lines})
    )


if __name__ == "__main__":
    main()
