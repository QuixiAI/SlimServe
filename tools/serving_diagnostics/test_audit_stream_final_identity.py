# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benign fixtures for streamed/terminal function-call correlation."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audit_stream_final_identity import audit


class IdentityAuditTests(unittest.TestCase):
    def audit_pair(self, stream, final):
        events = [
            {"type": "response.output_item.done", "output_index": 1, "item": stream},
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": [final]},
            },
        ]
        record = {
            "path": "/v1/responses",
            "status": 200,
            "request_body": {"input": [{"role": "user", "content": "HIDDEN_USER"}]},
            "response_body": "".join(
                "data: " + json.dumps(event) + "\n\n" for event in events
            ),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(json.dumps(record) + "\n")
            return audit([path])

    def test_incomplete_stream_call_hidden_by_fresh_terminal_call(self):
        stream = {
            "type": "function_call",
            "name": "test",
            "id": "item1",
            "call_id": "call1",
            "status": "incomplete",
            "arguments": '{"HIDDEN_CODE":',
        }
        final = {
            **stream,
            "id": "item2",
            "call_id": "call2",
            "status": "completed",
            "arguments": "{}",
        }
        result = self.audit_pair(stream, final)
        counts = result["counts"]
        for key in (
            "item_id_mismatch_pairs",
            "call_id_mismatch_pairs",
            "arguments_mismatch_pairs",
            "status_mismatch_pairs",
            "completed_responses_with_incomplete_stream_item",
        ):
            self.assertEqual(counts[key], 1)
        self.assertNotIn("HIDDEN_", json.dumps(result))
        self.assertEqual(
            result["argument_or_status_anomalies"][0]["terminal_status"], "completed"
        )

    def test_same_arguments_but_changed_ids_is_separate(self):
        stream = {
            "type": "function_call",
            "name": "test",
            "id": "item1",
            "call_id": "call1",
            "status": "completed",
            "arguments": "{}",
        }
        result = self.audit_pair(stream, {**stream, "id": "item2", "call_id": "call2"})
        self.assertEqual(result["counts"]["item_id_mismatch_pairs"], 1)
        self.assertEqual(result["counts"]["arguments_mismatch_pairs"], 0)
        self.assertEqual(result["argument_or_status_anomalies"], [])

    def test_different_function_names_are_not_assumed_to_match(self):
        stream = {
            "type": "function_call",
            "name": "first",
            "arguments": "{}",
            "status": "completed",
        }
        result = self.audit_pair(stream, {**stream, "name": "second"})
        self.assertEqual(
            result["counts"]["call_count_or_name_order_mismatch_responses"], 1
        )
        self.assertFalse(
            result["argument_or_status_anomalies"][0]["ordered_pairs_comparable"]
        )


if __name__ == "__main__":
    unittest.main()
