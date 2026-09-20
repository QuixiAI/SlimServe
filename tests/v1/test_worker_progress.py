# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for bounded worker diagnostics and exception transparency."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from vllm.utils import worker_progress as progress


def test_environment_gate_is_registered(monkeypatch, tmp_path):
    from vllm import envs

    getter = envs.environment_variables["VLLM_WORKER_PROGRESS_DIR"]
    monkeypatch.delenv("VLLM_WORKER_PROGRESS_DIR", raising=False)
    assert getter() is None
    monkeypatch.setenv("VLLM_WORKER_PROGRESS_DIR", str(tmp_path))
    assert getter() == str(tmp_path)
    assert "VLLM_WORKER_PROGRESS_DIR" not in envs.compile_factors()


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    instance = progress.Recorder(str(tmp_path), rank=3)
    monkeypatch.setattr(progress, "_recorder", instance)
    yield instance
    instance.close()


def test_unset_environment_does_not_initialize_or_touch_files(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_PROGRESS_DIR", raising=False)
    monkeypatch.setattr(progress, "_recorder", None)
    constructor = Mock(side_effect=AssertionError("must not initialize"))
    monkeypatch.setattr(progress, "Recorder", constructor)
    progress.initialize_worker_progress(0)
    with (
        progress.rpc_scope("sample_tokens"),
        progress.phase_scope(progress.Phase.SAMPLE),
    ):
        pass
    constructor.assert_not_called()


def test_nested_phases_and_exception_identity_are_preserved(recorder):
    failure = RuntimeError("SENTINEL_PRIVATE_EXCEPTION_PAYLOAD")
    with (
        pytest.raises(RuntimeError) as caught,
        progress.rpc_scope("sample_tokens"),
        progress.phase_scope(progress.Phase.PRIOR_OUTPUT),
    ):
        snapshot = progress.read_snapshot(recorder.path)
        assert snapshot["threads"][0]["active_phases"] == [
            "RPC_SAMPLE",
            "PRIOR_OUTPUT",
        ]
        raise failure
    assert caught.value is failure
    result = progress.read_snapshot(recorder.path)
    assert result["rank"] == 3
    assert result["threads"][0]["active_phases"] == []
    assert [r["edge"] for r in result["threads"][0]["records"]] == [
        "begin",
        "begin",
        "error",
        "error",
    ]
    assert b"SENTINEL_PRIVATE_EXCEPTION_PAYLOAD" not in recorder.path.read_bytes()


def test_fixed_ring_wraps_and_contains_no_rpc_payload(recorder):
    for _ in range(400):
        with progress.rpc_scope(b"SENTINEL_SERIALIZED_RPC_WITH_PRIVATE_PAYLOAD"):
            pass
    result = progress.read_snapshot(recorder.path)
    thread = result["threads"][0]
    assert thread["published_sequence"] == 800
    assert thread["history_truncated"]
    assert len(thread["records"]) == progress._CAPACITY
    assert thread["records"][-1]["operation"] == 400
    assert thread["active_phases"] == []
    assert recorder.path.stat().st_size == progress._FILE_SIZE
    assert recorder.path.stat().st_mode & 0o777 == 0o600
    assert b"PRIVATE_PAYLOAD" not in recorder.path.read_bytes()


def test_threads_publish_independent_bounded_histories(recorder):
    ready = threading.Barrier(3)
    leave = threading.Event()

    def write(phase):
        with progress.phase_scope(phase):
            ready.wait(timeout=5)
            leave.wait(timeout=5)

    threads = [
        threading.Thread(target=write, args=(progress.Phase.RPC_SAMPLE,)),
        threading.Thread(target=write, args=(progress.Phase.ASYNC_OUTPUT,)),
    ]
    for thread in threads:
        thread.start()
    try:
        ready.wait(timeout=5)
        result = progress.read_snapshot(recorder.path)
        assert {tuple(t["active_phases"]) for t in result["threads"]} == {
            ("RPC_SAMPLE",),
            ("ASYNC_OUTPUT",),
        }
        with progress.phase_scope(progress.Phase.SAMPLE):
            pass
        assert len(recorder._threads) == 2
        assert recorder.path.stat().st_size == progress._FILE_SIZE
    finally:
        leave.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert all(
        not t["active_phases"] for t in progress.read_snapshot(recorder.path)["threads"]
    )


def test_unpublished_record_does_not_replace_last_stable_phase(recorder):
    with progress.rpc_scope("sample_tokens"):
        before = progress.read_snapshot(recorder.path)
        slot = progress._HEADER_SIZE
        progress._U64.pack_into(recorder._map, slot + 64 + progress._RECORD.size, 2)
        assert progress.read_snapshot(recorder.path) == before


def test_diagnostic_write_failure_does_not_mask_serving_error(recorder):
    recorder._map.close()
    failure = ValueError("original serving error")
    with pytest.raises(ValueError) as caught, progress.rpc_scope("sample_tokens"):
        raise failure
    assert caught.value is failure
    assert not recorder.enabled


def test_reader_and_standalone_cli_limit_to_requested_pids(recorder, tmp_path):
    with progress.phase_scope(progress.Phase.MTP):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(progress.__file__)),
                "--directory",
                str(tmp_path),
                "--pids",
                str(os.getpid()),
                "1",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    snapshots = json.loads(result.stdout)
    assert [s["pid"] for s in snapshots] == [os.getpid(), 1]
    assert snapshots[0]["threads"][0]["active_phases"] == ["MTP"]
    assert snapshots[1]["error"] == "FileNotFoundError"


def test_invalid_file_does_not_break_reader(tmp_path):
    (tmp_path / "worker-7.bin").write_bytes(b"private malformed bytes")
    assert progress.snapshot_workers(str(tmp_path), [7]) == [
        {"pid": 7, "error": "ValueError"}
    ]


def test_initialization_failure_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_WORKER_PROGRESS_DIR", str(tmp_path))
    monkeypatch.setattr(progress, "_recorder", None)
    monkeypatch.setattr(
        progress, "Recorder", Mock(side_effect=PermissionError("private path"))
    )
    progress.initialize_worker_progress(1)
    assert progress._recorder is None


def test_real_sampler_prior_copy_hook_preserves_failure(recorder):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    failure = RuntimeError("synthetic copy wait failure")
    batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(),
        update_async_output_token_ids=Mock(side_effect=failure),
    )
    runner = SimpleNamespace(input_batch=batch)
    with pytest.raises(RuntimeError) as caught:
        GPUModelRunner._sample(runner, None, None)
    assert caught.value is failure
    records = progress.read_snapshot(recorder.path)["threads"][0]["records"]
    assert [(r["phase"], r["edge"]) for r in records] == [
        ("PRIOR_OUTPUT", "begin"),
        ("PRIOR_OUTPUT", "error"),
    ]


def test_real_async_output_hook_preserves_worker_failure_reply(recorder):
    from types import SimpleNamespace

    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.outputs import AsyncModelRunnerOutput

    class FailingOutput(AsyncModelRunnerOutput):
        def get_output(self):
            raise ValueError("synthetic materialization error")

    queue = Mock()
    worker = SimpleNamespace(worker_response_mq=queue)
    WorkerProc.enqueue_output(worker, FailingOutput())
    queue.enqueue.assert_called_once_with(
        (WorkerProc.ResponseStatus.FAILURE, "synthetic materialization error")
    )
    records = progress.read_snapshot(recorder.path)["threads"][0]["records"]
    assert [(r["phase"], r["edge"]) for r in records] == [
        ("ASYNC_OUTPUT", "begin"),
        ("ASYNC_OUTPUT", "error"),
    ]
