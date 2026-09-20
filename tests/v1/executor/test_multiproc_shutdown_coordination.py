# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.executor.multiproc_executor import MultiprocExecutor


def _monitor_executor(expected: bool):
    executor = object.__new__(MultiprocExecutor)
    sentinel = object()
    proc = SimpleNamespace(sentinel=sentinel, name="worker-0", exitcode=0)
    executor.workers = [SimpleNamespace(proc=proc)]
    executor._worker_exit_expected = threading.Event()
    if expected:
        executor.notify_shutdown_requested()
    executor.shutting_down = False
    executor.is_failed = False
    executor.failure_callback = MagicMock()
    executor.shutdown = MagicMock()
    return executor, sentinel


def test_monitor_suppresses_expected_worker_exit_without_skipping_cleanup():
    executor, sentinel = _monitor_executor(expected=True)

    with patch(
        "vllm.v1.executor.multiproc_executor.multiprocessing.connection.wait",
        return_value=[sentinel],
    ):
        executor.start_worker_monitor(inline=True)

    assert not executor.is_failed
    executor.shutdown.assert_not_called()
    executor.failure_callback.assert_not_called()
    assert not executor.shutting_down


def test_monitor_still_reports_unexpected_worker_exit():
    executor, sentinel = _monitor_executor(expected=False)
    failure_callback = executor.failure_callback

    with patch(
        "vllm.v1.executor.multiproc_executor.multiprocessing.connection.wait",
        return_value=[sentinel],
    ):
        executor.start_worker_monitor(inline=True)

    assert executor.is_failed
    executor.shutdown.assert_called_once_with()
    failure_callback.assert_called_once_with()
    assert executor.failure_callback is None


def _shutdown_executor(events, collective_rpc=None):
    executor = object.__new__(MultiprocExecutor)
    executor._worker_exit_expected = threading.Event()
    executor.shutting_down = False
    executor.is_failed = False
    executor.failure_callback = None

    death_writer = MagicMock()
    death_writer.close.side_effect = lambda: events.append("death-close")
    proc = MagicMock()
    proc.is_alive.return_value = False
    worker = SimpleNamespace(
        death_writer=death_writer,
        proc=proc,
        worker_response_mq=None,
    )
    executor.workers = [worker]
    executor.rpc_broadcast_mq = MagicMock()
    executor.response_mqs = []
    executor.collective_rpc = collective_rpc or MagicMock(
        side_effect=lambda *args, **kwargs: events.append("quiesced")
    )
    executor._ensure_worker_termination = MagicMock(
        side_effect=lambda *args, **kwargs: events.append("terminated")
    )
    return executor, death_writer


def test_shutdown_waits_for_quiesce_ack_before_death_pipe_close():
    events = []
    executor, death_writer = _shutdown_executor(events)

    executor.shutdown()

    executor.collective_rpc.assert_called_once()
    assert executor.collective_rpc.call_args.args == ("prepare_shutdown",)
    assert events == ["quiesced", "death-close", "terminated"]
    death_writer.close.assert_called_once_with()
    assert executor._worker_exit_expected.is_set()
    assert executor.shutting_down


def test_delayed_rank_ack_keeps_worker_processes_alive():
    events = []
    entered = threading.Event()
    release = threading.Event()

    def delayed_quiesce(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        events.append("quiesced")

    collective_rpc = MagicMock(side_effect=delayed_quiesce)
    executor, death_writer = _shutdown_executor(events, collective_rpc)
    thread = threading.Thread(target=executor.shutdown)
    thread.start()

    assert entered.wait(timeout=2)
    death_writer.close.assert_not_called()
    assert events == []

    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert events == ["quiesced", "death-close", "terminated"]


def test_shutdown_uses_one_shared_worker_grace_deadline():
    events = []
    executor, _ = _shutdown_executor(events)

    with (
        patch(
            "vllm.v1.executor.multiproc_executor.envs."
            "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
            120,
        ),
        patch(
            "vllm.v1.executor.multiproc_executor.time.monotonic",
            side_effect=[0.0, 30.0, 50.0],
        ),
    ):
        executor.shutdown()

    assert executor.collective_rpc.call_args.kwargs["timeout"] == 90.0
    assert executor._ensure_worker_termination.call_args.kwargs["timeout"] == 70.0
