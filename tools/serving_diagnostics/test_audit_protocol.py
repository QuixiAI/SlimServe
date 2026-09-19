# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only fixtures for aggregate capture auditing."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audit_protocol import audit, timestamp


def capture(events, *, truncated=False, ts=100, required=True, request_id="test"):
    return {
        "timestamp": ts,
        "request_id": request_id,
        "path": "/v1/responses",
        "status": 200,
        "elapsed_ms": 123.456,
        "request_body": {"tool_choice": "required" if required else "auto"},
        "response_headers": {"content-type": "text/event-stream"},
        "response_body": "".join("data: " + json.dumps(e) + "\n\n" for e in events),
        "response_capture_truncated": truncated,
    }


def terminal(status="completed", output=None):
    return {
        "type": f"response.{status}",
        "response": {
            "status": status,
            "output": output or [],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        },
    }


class AuditTests(unittest.TestCase):
    def run_audit(self, records, since=float("-inf"), until=float("inf")):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            return audit([path], since, until)

    def test_valid_empty_object_and_exact_totals(self):
        item = {"type": "function_call", "arguments": "{}", "status": "completed"}
        record = capture(
            [
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "x",
                    "delta": "{",
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "x",
                    "delta": "}",
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "x",
                    "arguments": "{}",
                },
                {"type": "response.output_item.done", "item": item},
                terminal(output=[item]),
            ]
        )
        result = self.run_audit([record])
        self.assertEqual(result["terminal_completed"], 1)
        self.assertEqual(result["required_completed_without_calls"], 0)
        self.assertEqual(result["argument_delta_done_mismatches"], 0)
        self.assertEqual(result["invalid_argument_done_json"], 0)
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (12, 7))
        self.assertEqual(result["elapsed_ms"], 123.456)

    def test_capture_truncation_separate_from_missing_terminal(self):
        result = self.run_audit([capture([], truncated=True), capture([])])
        self.assertEqual(result["capture_truncated_without_terminal"], 1)
        self.assertEqual(result["missing_terminal_without_capture_truncation"], 1)
        self.assertEqual(result["missing_usage_records"], 2)

    def test_invalid_done_final_and_mismatch(self):
        item = {"type": "function_call", "arguments": '{"x":', "status": "completed"}
        result = self.run_audit(
            [
                capture(
                    [
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": "x",
                            "delta": "{}",
                        },
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": "x",
                            "arguments": '{"x":',
                        },
                        {"type": "response.output_item.done", "item": item},
                        terminal(output=[item]),
                    ]
                )
            ]
        )
        self.assertEqual(result["invalid_argument_done_json"], 1)
        self.assertEqual(result["invalid_final_argument_json"], 1)
        self.assertEqual(result["completed_invalid_final_argument_json"], 1)
        self.assertEqual(result["argument_delta_done_mismatches"], 1)

    def test_incomplete_partial_args_are_not_completed_invalid(self):
        item = {"type": "function_call", "arguments": "{", "status": "incomplete"}
        result = self.run_audit([capture([terminal("incomplete", [item])])])
        self.assertEqual(result["terminal_incomplete"], 1)
        self.assertEqual(result["invalid_final_argument_json"], 1)
        self.assertEqual(result["completed_invalid_final_argument_json"], 0)
        self.assertEqual(result["required_completed_without_calls"], 0)

    def test_required_no_call_and_failed(self):
        result = self.run_audit(
            [
                capture([terminal()]),
                capture([terminal()], required=False),
                capture([terminal("failed")]),
                capture(
                    [
                        terminal(
                            output=[{"type": "custom_tool_call", "input": "freeform"}]
                        )
                    ]
                ),
            ]
        )
        self.assertEqual(result["required_completed_without_calls"], 1)
        self.assertEqual(result["terminal_failed"], 1)
        self.assertEqual(result["terminal_completed"], 3)

    def test_until_is_exclusive(self):
        result = self.run_audit(
            [capture([terminal()], ts=99), capture([terminal()], ts=100)],
            since=99,
            until=100,
        )
        self.assertEqual(result["finished_requests"], 1)
        self.assertEqual(result["input_tokens"], 12)

    def test_since_and_pending_start(self):
        result = self.run_audit(
            [
                {
                    "event": "request_start",
                    "timestamp": 100,
                    "request_id": "test",
                    "path": "/v1/responses",
                },
                capture([terminal()], ts=100),
                {
                    "event": "request_start",
                    "timestamp": 101,
                    "request_id": "pending",
                    "path": "/v1/responses",
                },
                capture([terminal()], ts=90),
                {
                    "timestamp": 101,
                    "request_id": "health",
                    "path": "/health",
                    "status": 502,
                },
            ],
            since=100,
        )
        self.assertEqual(result["pending_or_missing_finish_records"], 1)
        self.assertEqual(result["finished_requests"], 1)
        self.assertEqual(result["ignored_non_generation_records"], 1)
        self.assertEqual(result["http_errors"], 0)
        self.assertEqual(timestamp("1970-01-01T00:01:40Z"), 100)
        self.assertEqual(timestamp("100"), 100)


if __name__ == "__main__":
    unittest.main()
