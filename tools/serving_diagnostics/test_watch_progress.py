# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benign deterministic watchdog fixtures; no services or wall-clock waits."""

import argparse
import json
import math
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from watch_progress import (
    MAX_STACK_TARGETS,
    Metrics,
    ProgressDetector,
    diagnostic,
    local_metrics_url,
    parse_metrics,
    prioritized_stack_targets,
    private_output,
    process_snapshot,
    safe_journal_frames,
    safe_spy_threads,
    service_main_pid,
    stack_snapshot,
)


class DetectorTests(unittest.TestCase):
    def test_idle_never_stalls(self):
        detector = ProgressDetector()
        for now in (0, 45, 300):
            self.assertEqual(detector.update(now, Metrics(0, 0, 0))["state"], "idle")

    def test_first_active_sample_does_not_infer_past_stall(self):
        detector = ProgressDetector()
        result = detector.update(300, Metrics(10, 20, 1))
        self.assertEqual(result["state"], "waiting")
        self.assertFalse(result["diagnose"])

    def test_stall_fires_once_until_token_recovery(self):
        detector = ProgressDetector()
        sample = Metrics(10, 20, 1)
        detector.update(0, sample)
        self.assertFalse(detector.update(44, sample)["diagnose"])
        self.assertTrue(detector.update(45, sample)["diagnose"])
        self.assertEqual(detector.update(90, sample)["state"], "stall_ongoing")
        self.assertEqual(detector.update(91, Metrics(11, 20, 1))["state"], "recovered")
        self.assertTrue(detector.update(136, Metrics(11, 20, 1))["diagnose"])

    def test_prompt_progress_resets_window(self):
        detector = ProgressDetector()
        detector.update(0, Metrics(10, 20, 1))
        self.assertEqual(detector.update(44, Metrics(10, 21, 1))["state"], "progress")
        self.assertFalse(detector.update(45, Metrics(10, 21, 1))["diagnose"])
        self.assertTrue(detector.update(89, Metrics(10, 21, 1))["diagnose"])

    def test_counter_reset_rearms_stall(self):
        detector = ProgressDetector()
        detector.update(0, Metrics(10, 20, 1))
        detector.update(45, Metrics(10, 20, 1))
        result = detector.update(46, Metrics(0, 0, 1))
        self.assertEqual(result["state"], "counter_reset")
        self.assertFalse(result["diagnose"])
        self.assertTrue(detector.update(91, Metrics(0, 0, 1))["diagnose"])

    def test_creation_epoch_detects_zero_to_zero_restart(self):
        detector = ProgressDetector()
        detector.update(0, Metrics(0, 0, 1, 100, 100))
        detector.update(45, Metrics(0, 0, 1, 100, 100))
        result = detector.update(46, Metrics(0, 0, 1, 200, 200))
        self.assertEqual(result["state"], "counter_reset")
        self.assertEqual(result["reset_evidence"], "creation_epoch_changed")
        self.assertTrue(detector.update(91, Metrics(0, 0, 1, 200, 200))["diagnose"])

    def test_unavailable_metrics_are_not_stall_evidence(self):
        detector = ProgressDetector()
        sample = Metrics(10, 20, 1)
        detector.update(0, sample)
        self.assertEqual(detector.update(40, None)["state"], "metrics_unavailable")
        self.assertFalse(detector.update(100, sample)["diagnose"])
        self.assertFalse(detector.update(144, sample)["diagnose"])
        self.assertTrue(detector.update(145, sample)["diagnose"])

    def test_unavailability_does_not_duplicate_existing_stall(self):
        detector = ProgressDetector()
        sample = Metrics(10, 20, 1)
        detector.update(0, sample)
        detector.update(45, sample)
        detector.update(50, None)
        self.assertFalse(detector.update(100, sample)["diagnose"])
        self.assertFalse(detector.update(200, sample)["diagnose"])
        self.assertEqual(detector.update(201, Metrics(11, 20, 1))["state"], "recovered")

    def test_idle_recovery_rearms_from_new_running_sample(self):
        detector = ProgressDetector()
        detector.update(0, Metrics(10, 20, 1))
        detector.update(45, Metrics(10, 20, 1))
        self.assertEqual(
            detector.update(46, Metrics(10, 20, 0))["state"], "recovered_idle"
        )
        self.assertFalse(detector.update(300, Metrics(10, 20, 1))["diagnose"])
        self.assertTrue(detector.update(345, Metrics(10, 20, 1))["diagnose"])


class DiagnosticTests(unittest.TestCase):
    def test_metrics_aggregate_without_retaining_labels(self):
        text = """# benign fixture
vllm:generation_tokens_total{model_name="HIDDEN_LABEL"} 10
vllm:generation_tokens_total{model_name="other"} 2
vllm:prompt_tokens_total 20
vllm:num_requests_running 1
unrelated_metric{label="HIDDEN_OTHER"} 999
"""
        self.assertEqual(parse_metrics(text), Metrics(12, 20, 1))
        self.assertNotIn("HIDDEN", repr(parse_metrics(text)))
        epochs = parse_metrics(text + "vllm:generation_tokens_created 100\n")
        self.assertEqual(epochs.generation_created_min, 100)
        self.assertEqual(epochs.generation_created_max, 100)

    def test_missing_or_invalid_metrics_are_unavailable_not_zero(self):
        for text in (
            "",
            "vllm:generation_tokens_total 1",
            "vllm:generation_tokens_total NaN",
            "vllm:generation_tokens_total -1",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_metrics(text)

    def test_metrics_url_only_accepts_plain_loopback(self):
        for url in ("http://127.0.0.1:8001/metrics", "http://[::1]:8001/metrics"):
            self.assertEqual(local_metrics_url(url), url)
        for url in (
            "https://127.0.0.1/metrics",
            "http://example.com/metrics",
            "http://user:secret@127.0.0.1/metrics",
            "http://127.0.0.1/metrics?secret=1",
        ):
            with self.subTest(url=url), self.assertRaises(argparse.ArgumentTypeError):
                local_metrics_url(url)

    def test_journal_keeps_frames_and_types_but_no_messages_or_source(self):
        text = """request: HIDDEN_PAYLOAD
  File "/private/path/worker.py", line 10, in sample_tokens
    result = execute("HIDDEN_SOURCE")
RuntimeError: HIDDEN_EXCEPTION_MESSAGE
sample_tokens RPC timed out; HIDDEN_RPC_DATA
"""
        frames = safe_journal_frames(text)
        self.assertEqual(
            frames[0], {"file": "worker.py", "line": 10, "function": "sample_tokens"}
        )
        self.assertIn({"exception_type": "RuntimeError"}, frames)
        self.assertIn({"category": "sample_tokens_timeout"}, frames)
        self.assertNotIn("HIDDEN", json.dumps(frames))
        self.assertNotIn("/private", json.dumps(frames))

    def test_stack_dump_excludes_command_header_and_locals(self):
        text = json.dumps(
            [
                {
                    "thread_id": 123,
                    "active": True,
                    "thread_name": "HIDDEN_NAME",
                    "process_info": {"cmdline": "HIDDEN_KEY"},
                    "frames": [
                        {
                            "name": "sample_tokens",
                            "filename": "/private/worker.py",
                            "line": 12,
                            "locals": {"input": "HIDDEN_LOCAL"},
                        }
                    ],
                }
            ]
        )
        self.assertEqual(
            safe_spy_threads(text),
            [
                {
                    "thread_id": 123,
                    "active": True,
                    "frames": [
                        {"function": "sample_tokens", "file": "worker.py", "line": 12}
                    ],
                }
            ],
        )
        for sudo in (False, True):
            with (
                self.subTest(sudo=sudo),
                patch(
                    "watch_progress.run_readonly",
                    return_value=(text, {"available": True}),
                ) as run,
            ):
                result = stack_snapshot("/fixture/py-spy", [123], sudo=sudo)
            argv = ["/fixture/py-spy", "dump", "--native", "--json", "--pid", "123"]
            self.assertEqual(
                run.call_args.args[0], ["sudo", "-n", *argv] if sudo else argv
            )
            self.assertNotIn("--locals", run.call_args.args[0])
            self.assertNotIn("HIDDEN", json.dumps(result))

    def test_missing_stack_tool_records_failure(self):
        with patch(
            "watch_progress.run_readonly",
            return_value=(
                None,
                {"available": False, "error_type": "FileNotFoundError"},
            ),
        ):
            result = stack_snapshot("/fixture/missing", [123])
        self.assertFalse(result[0]["available"])
        self.assertEqual(result[0]["error_type"], "FileNotFoundError")

    def test_process_tree_selects_descendants_without_command_arguments(self):
        output = (
            "1 0 Ss init wait\n10 1 Sl python futex_wait\n"
            "11 10 Sl worker do_poll\n12 1 S unrelated wait\n"
        )
        with patch(
            "watch_progress.run_readonly", return_value=(output, {"available": True})
        ) as run:
            result = process_snapshot([10])
        self.assertEqual([row["pid"] for row in result["processes"]], [10, 11])
        self.assertNotIn("args", run.call_args.args[0][-1])

    def test_tp4_workers_are_captured_before_core_api_and_helpers(self):
        rows = [
            {"pid": 10, "name": "python"},
            {"pid": 11, "name": "python"},
            {"pid": 12, "name": "VLLM::EngineCore"},
            *[
                {"pid": 20 + rank, "name": f"VLLM::Worker_TP{rank}"}
                for rank in range(4)
            ],
            *[{"pid": 30 + rank, "name": "helper"} for rank in range(10)],
        ]
        targets = prioritized_stack_targets(rows, [10])
        self.assertEqual(targets[:6], [20, 21, 22, 23, 12, 10])
        with patch(
            "watch_progress.run_readonly", return_value=("[]", {"available": True})
        ) as run:
            stacks = stack_snapshot("/fixture/py-spy", targets)
        self.assertEqual(len(stacks), MAX_STACK_TARGETS)
        self.assertEqual([stack["pid"] for stack in stacks[:6]], targets[:6])
        self.assertEqual(run.call_count, MAX_STACK_TARGETS)

    def test_service_main_pid_is_rediscovered_for_each_diagnostic(self):
        args = SimpleNamespace(
            pid=[],
            service_unit="fixture.service",
            journal_unit=[],
            system_journal=False,
            py_spy="/fixture/py-spy",
            sudo_py_spy=True,
        )
        with (
            patch(
                "watch_progress.service_main_pid",
                side_effect=[
                    ([10], {"available": True, "main_pid": 10}),
                    ([20], {"available": True, "main_pid": 20}),
                ],
            ),
            patch(
                "watch_progress.process_snapshot",
                side_effect=[
                    {"processes": [{"pid": 10, "name": "python"}]},
                    {"processes": [{"pid": 20, "name": "python"}]},
                ],
            ) as tree,
            patch("watch_progress.journal_snapshot", return_value={}),
            patch("watch_progress.stack_snapshot", return_value=[]) as stacks,
        ):
            first = diagnostic(args, math.inf)
            second = diagnostic(args, math.inf)
        self.assertEqual([call.args[0] for call in tree.call_args_list], [[10], [20]])
        self.assertEqual([call.args[1] for call in stacks.call_args_list], [[10], [20]])
        self.assertEqual(first["service_discovery"]["main_pid"], 10)
        self.assertEqual(second["service_discovery"]["main_pid"], 20)

    def test_inactive_service_never_falls_back_to_global_processes(self):
        args = SimpleNamespace(
            pid=[],
            service_unit="fixture.service",
            journal_unit=[],
            system_journal=False,
            py_spy="/fixture/py-spy",
            sudo_py_spy=False,
        )
        with (
            patch(
                "watch_progress.service_main_pid",
                return_value=(
                    [],
                    {"available": False, "error_type": "service_inactive"},
                ),
            ),
            patch("watch_progress.process_snapshot") as tree,
            patch("watch_progress.journal_snapshot", return_value={}),
        ):
            result = diagnostic(args, math.inf)
        tree.assert_not_called()
        self.assertEqual(
            result["python_native_stacks"][0]["error_type"], "no_target_pid"
        )

    def test_service_lookup_is_read_only_and_requires_positive_pid(self):
        for output, expected in (("123\n", [123]), ("0\n", []), ("invalid\n", [])):
            with (
                self.subTest(output=output),
                patch(
                    "watch_progress.run_readonly",
                    return_value=(output, {"available": True}),
                ) as run,
            ):
                pids, _ = service_main_pid("fixture.service")
            self.assertEqual(pids, expected)
            self.assertEqual(
                run.call_args.args[0],
                [
                    "systemctl",
                    "--user",
                    "show",
                    "--property=MainPID",
                    "--value",
                    "--",
                    "fixture.service",
                ],
            )

    def test_native_unknown_source_and_late_main_thread_survive_caps(self):
        traces = [
            {"pid": 1, "thread_id": index, "os_thread_id": index + 10, "frames": []}
            for index in range(140)
        ]
        traces.append(
            {
                "pid": 1,
                "thread_id": 999,
                "os_thread_id": 1,
                "frames": [{"name": "native_wait", "filename": None, "line": None}],
            }
        )
        threads = safe_spy_threads(json.dumps(traces))
        self.assertEqual(len(threads), 128)
        self.assertTrue(threads[0]["main_thread"])
        self.assertEqual(
            threads[0]["frames"],
            [{"function": "native_wait", "file": None, "line": None}],
        )

    def test_output_is_private_append_only_and_refuses_symlink(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "watch.jsonl"
            for value in ("first", "second"):
                with private_output(path) as output:
                    output.write(value + "\n")
            self.assertEqual(path.read_text(), "first\nsecond\n")
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            link = Path(directory) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                private_output(link)


if __name__ == "__main__":
    unittest.main()
