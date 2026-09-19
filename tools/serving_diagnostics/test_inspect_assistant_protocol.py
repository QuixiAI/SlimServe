# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benign CPU fixtures for payload-free assistant protocol inspection."""

import json
import unittest

from inspect_assistant_protocol import arguments_summary, inspect_record


class InspectorTests(unittest.TestCase):
    def test_history_filters_payloads_and_keeps_empty_message_and_json_offset(self):
        record = {
            "request_id": "request_test",
            "status": 400,
            "request_body": {
                "input": [
                    {"role": "system", "content": "SYSTEM_SECRET"},
                    {"role": "user", "content": "USER_SECRET"},
                    {"type": "function_call_output", "output": "TOOL_SECRET"},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "id": "msg_empty",
                    },
                    {
                        "type": "function_call",
                        "id": "fc_test",
                        "call_id": "call_test",
                        "arguments": '{"snippet":',
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "GENERATED_CODE"}],
                    },
                ]
            },
            "response_body": {
                "error": {
                    "message": "Expecting value: line 1 column 12 (char 11) USER_SECRET"
                }
            },
        }
        result = inspect_record(record)
        serialized = json.dumps(result)
        for hidden in (
            "SYSTEM_SECRET",
            "USER_SECRET",
            "TOOL_SECRET",
            "GENERATED_CODE",
            "snippet",
        ):
            self.assertNotIn(hidden, serialized)
        self.assertEqual(
            [x["position"] for x in result["assistant_history"]], [3, 4, 5]
        )
        self.assertEqual(result["assistant_history"][0]["content_part_count"], 0)
        args = result["assistant_history"][1]["arguments"]
        self.assertFalse(args["valid_json"])
        self.assertEqual(args["error_pos"], 11)
        self.assertEqual(result["response"]["error"]["json_error_pos"], 11)

    def stream_result(self, delta, done, item_done, final, truncated=False):
        item = {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "arguments": final,
            "status": "completed",
        }
        events = [
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": delta[:2],
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_test",
                "delta": delta[2:],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_test",
                "arguments": done,
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {**item, "arguments": item_done},
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "id": "resp_test",
                    "output": [item],
                },
            },
        ]
        return inspect_record(
            {
                "status": 200,
                "request_body": {},
                "response_capture_truncated": truncated,
                "response_headers": {"content-type": "text/event-stream"},
                "response_body": "".join(
                    "data: " + json.dumps(event) + "\n\n" for event in events
                ),
            }
        )

    def test_valid_stream_exact_matches_all_stages(self):
        result = self.stream_result('{"x":1}', '{"x":1}', '{"x":1}', '{"x":1}')
        call = result["response"]["stream_calls"][0]
        self.assertEqual(len(result["response"]["stream_calls"]), 1)
        self.assertEqual(call["delta_events"], 2)
        self.assertTrue(call["delta"]["valid_json"])
        self.assertTrue(call["delta_vs_done"]["equal"])
        self.assertTrue(call["delta_vs_item_done"]["equal"])
        self.assertTrue(call["delta_vs_final"]["equal"])
        self.assertTrue(call["done_vs_final"]["equal"])
        self.assertTrue(call["item_done_vs_final"]["equal"])

    def test_mismatch_reports_exact_position_without_arguments(self):
        result = self.stream_result('{"x":1}', '{"x":2}', '{"x":3}', '{"x":', True)
        call = result["response"]["stream_calls"][0]
        self.assertEqual(call["delta_vs_done"]["first_difference_char"], 5)
        self.assertEqual(call["delta_vs_final"]["first_difference_char"], 5)
        self.assertFalse(call["final"]["valid_json"])
        self.assertTrue(result["capture_truncated"])
        self.assertNotIn('"x"', json.dumps(result))

    def test_nonfinite_json_rejected_but_empty_object_allowed(self):
        for value in ("NaN", "Infinity", '{"n":-Infinity}'):
            self.assertFalse(arguments_summary(value)["valid_json"])
        for value in ("{}", "[]", "null", '"hello"'):
            self.assertTrue(arguments_summary(value)["valid_json"])

    def test_chat_history_only_assistant_tool_arguments(self):
        result = inspect_record(
            {
                "request_body": {
                    "messages": [
                        {"role": "tool", "content": "HIDE_TOOL"},
                        {
                            "role": "assistant",
                            "content": "HIDE_ASSISTANT_CODE",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "test", "arguments": "{}"},
                                },
                            ],
                        },
                    ]
                },
                "response_body": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "HIDE_RESPONSE_CODE",
                            }
                        }
                    ]
                },
            }
        )
        self.assertEqual(len(result["assistant_history"]), 1)
        self.assertTrue(
            result["assistant_history"][0]["tool_calls"][0]["arguments"]["valid_json"]
        )
        self.assertNotIn("HIDE_", json.dumps(result))

    def test_invalid_sse_and_missing_terminal_retained_as_diagnostics(self):
        result = inspect_record(
            {"response_body": "data: {\n\n", "response_capture_truncated": True}
        )
        self.assertEqual(result["response"]["invalid_sse_events"], 1)
        self.assertEqual(result["response"]["terminal_events"], [])
        self.assertTrue(result["capture_truncated"])

    def test_finished_reasoning_only_stream_is_unterminated_despite_http200(self):
        events = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"created_at": 100},
            },
            {
                "type": "response.reasoning_text.delta",
                "sequence_number": 1,
                "delta": "DO_NOT_SHOW",
            },
        ]
        result = inspect_record(
            {
                "status": 200,
                "elapsed_ms": 59000,
                "response_headers": {
                    "content-type": "text/event-stream; charset=utf-8",
                    "authorization": "DO_NOT_SHOW",
                },
                "response_body": "".join(
                    "data: " + json.dumps(event) + "\n\n" for event in events
                ),
            }
        )
        response = result["response"]
        self.assertTrue(response["unterminated_finished_stream"])
        self.assertTrue(response["missing_terminal_without_capture_truncation"])
        self.assertEqual(response["event_counts"]["response.reasoning_text.delta"], 1)
        self.assertEqual(response["first_events"][0]["response_created_at"], 100)
        self.assertEqual(result["elapsed_ms"], 59000)
        self.assertIsNone(response["first_chunk_ms"])
        self.assertNotIn("DO_NOT_SHOW", json.dumps(result))
        self.assertNotIn("authorization", json.dumps(result))

    def test_active_capture_is_not_reported_as_unterminated_finish(self):
        result = inspect_record(
            {
                "event": "request_start",
                "response_headers": {"content-type": "text/event-stream"},
                "response_body": "",
            }
        )
        self.assertFalse(result["response"]["unterminated_finished_stream"])

    def test_chat_done_is_terminal_for_unterminated_filter(self):
        result = inspect_record({"status": 200, "response_body": "data: [DONE]\n\n"})
        self.assertFalse(result["response"]["unterminated_finished_stream"])


if __name__ == "__main__":
    unittest.main()
