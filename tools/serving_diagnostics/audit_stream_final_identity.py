#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare streamed and terminal function calls without displaying payload text.

Pairs calls by output order only when function counts/names match in that order.
Identity mismatches are reported separately from argument byte mismatches.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_protocol import TERMINALS, sse_events, timestamp
from capture_paths import DEFAULT_LOG_PATH
from inspect_assistant_protocol import arguments_summary, identifier


def call_summary(item):
    return {
        "id": identifier(item.get("id")),
        "call_id": identifier(item.get("call_id")),
        "status": item.get("status"),
        "arguments": arguments_summary(item.get("arguments")),
    }


def audit(paths, since=float("-inf")):
    counts = Counter()
    anomalies = []
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
                    or record.get("event") == "request_start"
                ):
                    continue
                if (
                    record.get("timestamp", 0) < since
                    or record.get("path") != "/v1/responses"
                ):
                    continue
                body = record.get("response_body")
                if not isinstance(body, str):
                    continue
                counts["finished_response_streams"] += 1
                stream_calls = []
                final = None
                for event in sse_events(body, counts):
                    if event.get("type") == "response.output_item.done":
                        item = event.get("item", {})
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "function_call"
                        ):
                            stream_calls.append((event.get("output_index"), item))
                    if event.get("type") in TERMINALS:
                        final = event.get("response")
                if not isinstance(final, dict):
                    counts["without_terminal_response"] += 1
                    continue
                if final.get("status") != "completed":
                    counts["noncompleted_terminal_response"] += 1
                    continue
                counts["completed_responses"] += 1
                if all(type(index) is int for index, _ in stream_calls):
                    stream_calls.sort(key=lambda pair: pair[0])
                streamed = [item for _, item in stream_calls]
                terminal = [
                    item
                    for item in final.get("output", [])
                    if isinstance(item, dict) and item.get("type") == "function_call"
                ]
                if not streamed and not terminal:
                    continue
                counts["completed_responses_with_function_calls"] += 1
                counts["stream_item_done_calls"] += len(streamed)
                counts["terminal_calls"] += len(terminal)
                incomplete = [
                    item for item in streamed if item.get("status") != "completed"
                ]
                invalid = [
                    item
                    for item in streamed
                    if not arguments_summary(item.get("arguments"))["valid_json"]
                ]
                counts["stream_items_not_completed"] += len(incomplete)
                counts["stream_items_invalid_json"] += len(invalid)
                counts["completed_responses_with_incomplete_stream_item"] += bool(
                    incomplete
                )
                counts["completed_responses_with_invalid_stream_item"] += bool(invalid)
                alignment = len(streamed) == len(terminal) and all(
                    a.get("name") == b.get("name") for a, b in zip(streamed, terminal)
                )
                if not alignment:
                    counts["call_count_or_name_order_mismatch_responses"] += 1
                    pairs = []
                else:
                    pairs = list(zip(streamed, terminal))
                    counts["order_aligned_responses"] += 1
                comparisons = []
                for left, right in pairs:
                    result = {
                        "item_id_equal": left.get("id") == right.get("id"),
                        "call_id_equal": left.get("call_id") == right.get("call_id"),
                        "arguments_equal": left.get("arguments")
                        == right.get("arguments"),
                        "status_equal": left.get("status") == right.get("status"),
                    }
                    comparisons.append(result)
                    counts["order_aligned_call_pairs"] += 1
                    for field, equal in result.items():
                        counts[
                            field.removesuffix("_equal") + "_mismatch_pairs"
                        ] += not equal
                if comparisons:
                    counts["identity_mismatch_responses"] += any(
                        not c["item_id_equal"] or not c["call_id_equal"]
                        for c in comparisons
                    )
                    counts["arguments_mismatch_responses"] += any(
                        not c["arguments_equal"] for c in comparisons
                    )
                if (
                    incomplete
                    or invalid
                    or not alignment
                    or any(not c["arguments_equal"] for c in comparisons)
                ):
                    anomalies.append(
                        {
                            "request_id": identifier(record.get("request_id")),
                            "timestamp": record.get("timestamp"),
                            "capture_truncated": bool(
                                record.get("response_capture_truncated")
                            ),
                            "terminal_status": final.get("status"),
                            "ordered_pairs_comparable": alignment,
                            "comparisons": comparisons,
                            "stream_calls": [call_summary(item) for item in streamed],
                            "terminal_calls": [call_summary(item) for item in terminal],
                        }
                    )
    return {"counts": dict(counts), "argument_or_status_anomalies": anomalies}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_LOG_PATH],
    )
    parser.add_argument("--since", type=timestamp, default=float("-inf"))
    args = parser.parse_args()
    print(json.dumps(audit(args.paths, args.since), indent=2))
