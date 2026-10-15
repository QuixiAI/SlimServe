# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shutdown coordination between MultiprocExecutor and its worker monitor.

Runs on CPU: worker processes are stand-ins, and the response queues in the
cancellation tests are real shared-memory MessageQueues used in-process.
"""

import signal
import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

_WAIT = "vllm.v1.executor.multiproc_executor.multiprocessing.connection.wait"
_PLATFORM = "vllm.v1.executor.multiproc_executor.current_platform"


def _bare_executor() -> MultiprocExecutor:
    executor = object.__new__(MultiprocExecutor)
    executor._worker_exit_expected = threading.Event()
    executor._response_queues_cancelled = False
    executor._quiesce_in_progress = False
    executor.shutting_down = False
    executor.is_failed = False
    executor.failure_callback = None
    executor.response_mqs = []
    return executor


def _monitor_executor(expected: bool, exitcode: int = 0):
    executor = _bare_executor()
    sentinel = object()
    proc = SimpleNamespace(sentinel=sentinel, name="worker-0", exitcode=exitcode)
    executor.workers = [SimpleNamespace(proc=proc)]
    if expected:
        executor.notify_shutdown_requested()
    executor.failure_callback = MagicMock()
    executor.shutdown = MagicMock()
    return executor, sentinel


@pytest.fixture
def on_xpu():
    with patch(_PLATFORM) as platform:
        platform.is_xpu.return_value = True
        yield


@pytest.fixture
def not_xpu():
    with patch(_PLATFORM) as platform:
        platform.is_xpu.return_value = False
        yield


def test_monitor_suppresses_report_but_cancels_reads_on_expected_exit():
    executor, sentinel = _monitor_executor(expected=True)
    mq = MagicMock()
    executor.response_mqs = [mq]

    with patch(_WAIT, return_value=[sentinel]):
        executor.start_worker_monitor(inline=True)

    assert not executor.is_failed
    executor.shutdown.assert_not_called()
    executor.failure_callback.assert_not_called()
    assert not executor.shutting_down
    # The engine may be blocked reading this rank: that read is cancelled,
    # and no new RPC is attempted against the dead rank.
    mq.shutdown.assert_called_once_with()
    assert executor._response_queues_cancelled
    executor.rpc_broadcast_mq = MagicMock()
    with pytest.raises(RuntimeError, match="shutting down"):
        executor.collective_rpc("execute_model")
    executor.rpc_broadcast_mq.enqueue.assert_not_called()


def test_monitor_still_reports_unexpected_worker_exit():
    executor, sentinel = _monitor_executor(expected=False)
    failure_callback = executor.failure_callback

    with patch(_WAIT, return_value=[sentinel]):
        executor.start_worker_monitor(inline=True)

    assert executor.is_failed
    executor.shutdown.assert_called_once_with()
    failure_callback.assert_called_once_with()
    assert executor.failure_callback is None


_LOGGER = "vllm.v1.executor.multiproc_executor.logger"


@pytest.mark.parametrize(
    "exitcode, abnormal",
    [
        (0, False),
        (-signal.SIGTERM, False),
        (-signal.SIGINT, False),
        (128 + signal.SIGTERM, False),
        (128 + signal.SIGINT, False),
        (1, True),
        (-signal.SIGSEGV, True),
        (-signal.SIGKILL, True),
        (128 + signal.SIGSEGV, True),
    ],
)
def test_monitor_logs_abnormal_exit_during_requested_shutdown(exitcode, abnormal):
    executor, sentinel = _monitor_executor(expected=True, exitcode=exitcode)
    mq = MagicMock()
    executor.response_mqs = [mq]

    with patch(_WAIT, return_value=[sentinel]), patch(_LOGGER) as logger:
        executor.start_worker_monitor(inline=True)

    # Either way the shutdown proceeds: no failure report, reads cancelled.
    assert not executor.is_failed
    executor.failure_callback.assert_not_called()
    executor.shutdown.assert_not_called()
    mq.shutdown.assert_called_once_with()
    if abnormal:
        logger.error.assert_called_once()
        assert exitcode in logger.error.call_args.args
    else:
        logger.error.assert_not_called()


def test_monitor_leaves_queues_to_shutdown_once_teardown_started():
    executor, sentinel = _monitor_executor(expected=True)
    executor.shutting_down = True
    mq = MagicMock()
    executor.response_mqs = [mq]

    with patch(_WAIT, return_value=[sentinel]):
        executor.start_worker_monitor(inline=True)

    mq.shutdown.assert_not_called()
    assert not executor.is_failed


def _shutdown_executor(events, collective_rpc=None, alive=True):
    executor = _bare_executor()

    death_writer = MagicMock()
    death_writer.close.side_effect = lambda: events.append("death-close")
    proc = MagicMock()
    proc.is_alive.return_value = alive
    worker = SimpleNamespace(
        death_writer=death_writer,
        proc=proc,
        worker_response_mq=None,
    )
    executor.workers = [worker]
    executor.rpc_broadcast_mq = MagicMock()
    executor.collective_rpc = collective_rpc or MagicMock(
        side_effect=lambda *args, **kwargs: events.append("quiesced")
    )
    executor._ensure_worker_termination = MagicMock(
        side_effect=lambda *args, **kwargs: events.append("terminated")
    )
    return executor, death_writer


def test_shutdown_waits_for_quiesce_ack_before_death_pipe_close(on_xpu):
    events = []
    executor, death_writer = _shutdown_executor(events)

    executor.shutdown()

    executor.collective_rpc.assert_called_once()
    assert executor.collective_rpc.call_args.args == ("prepare_shutdown",)
    assert events == ["quiesced", "death-close", "terminated"]
    death_writer.close.assert_called_once_with()
    assert executor._worker_exit_expected.is_set()
    assert executor.shutting_down
    assert not executor._quiesce_in_progress


def test_non_xpu_shutdown_sends_no_quiesce_rpc(not_xpu):
    events = []
    executor, _ = _shutdown_executor(events)

    executor.shutdown()

    executor.collective_rpc.assert_not_called()
    assert events == ["death-close", "terminated"]


def test_shutdown_skips_quiesce_when_a_rank_is_already_dead(on_xpu):
    events = []
    executor, _ = _shutdown_executor(events, alive=False)

    executor.shutdown()

    executor.collective_rpc.assert_not_called()
    assert events == ["death-close", "terminated"]


def test_shutdown_skips_quiesce_after_queues_were_cancelled(on_xpu):
    events = []
    executor, _ = _shutdown_executor(events)
    executor._response_queues_cancelled = True

    executor.shutdown()

    executor.collective_rpc.assert_not_called()
    assert events == ["death-close", "terminated"]


def test_delayed_rank_ack_keeps_worker_processes_alive(on_xpu):
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


def test_shutdown_uses_one_shared_worker_grace_deadline(on_xpu):
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


# ---------------------------------------------------------------------------
# Real response queue: a rank exits while the engine is blocked reading it.
# ---------------------------------------------------------------------------


# Queues stay referenced for the whole session: collecting one mid-run lets
# its zmq Context terminate while sibling sockets are still open, which blocks.
_LIVE_QUEUES: list[MessageQueue] = []


@pytest.fixture
def response_queue():
    """(worker-side writer, engine-side reader) of one response queue."""
    writer = MessageQueue(1, 1)
    reader = MessageQueue.create_from_handle(writer.export_handle(), 0)
    ready = threading.Thread(target=writer.wait_until_ready)
    ready.start()
    reader.wait_until_ready()
    ready.join(timeout=5)
    _LIVE_QUEUES.extend((writer, reader))
    yield writer, reader
    reader.shutdown()


def _rpc_executor(reader):
    executor = _bare_executor()
    executor.rpc_broadcast_mq = MagicMock()
    executor.response_mqs = [reader]
    executor.futures_queue = deque()
    sentinel = object()
    proc = SimpleNamespace(sentinel=sentinel, name="worker-0", exitcode=-15)
    executor.workers = [SimpleNamespace(proc=proc)]
    executor.failure_callback = MagicMock()
    return executor, sentinel


def _rpc_in_thread(executor, method, **kwargs):
    outcome = {}

    def call():
        start = time.monotonic()
        try:
            outcome["result"] = executor.collective_rpc(method, **kwargs)
        except Exception as e:  # noqa: BLE001 - the test inspects it
            outcome["error"] = e
        outcome["elapsed"] = time.monotonic() - start

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    return thread, outcome


def test_live_rank_reply_is_delivered(response_queue):
    writer, reader = response_queue
    executor, _ = _rpc_executor(reader)
    thread, outcome = _rpc_in_thread(executor, "execute_model")

    writer.enqueue((WorkerProc.ResponseStatus.SUCCESS, "output"))

    thread.join(timeout=5)
    assert outcome.get("result") == ["output"]


def test_rank_exit_during_execute_model_cancels_the_blocked_read(response_queue):
    """SIGTERM reaches the engine and the workers together: the engine is
    blocked in execute_model's dequeue on a rank that has just died."""
    _, reader = response_queue
    executor, sentinel = _rpc_executor(reader)
    thread, outcome = _rpc_in_thread(executor, "execute_model")
    time.sleep(0.3)
    assert thread.is_alive(), "the dequeue should be blocked on the dead rank"

    executor.notify_shutdown_requested()  # EngineCore signal handler
    with patch(_WAIT, return_value=[sentinel]):
        executor.start_worker_monitor(inline=True)

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), RuntimeError)
    assert outcome["elapsed"] < 5
    # Expected exit: no failure report, and the later shutdown still runs.
    assert not executor.is_failed
    executor.failure_callback.assert_not_called()
    assert not executor.shutting_down


def test_rank_exit_during_quiesce_does_not_wait_for_the_grace_period(
    response_queue, on_xpu
):
    _, reader = response_queue
    executor, sentinel = _rpc_executor(reader)
    executor.notify_shutdown_requested()
    executor.shutting_down = True
    executor._quiesce_in_progress = True
    thread, outcome = _rpc_in_thread(executor, "prepare_shutdown", timeout=60)
    time.sleep(0.3)
    assert thread.is_alive()

    with patch(_WAIT, return_value=[sentinel]):
        executor.start_worker_monitor(inline=True)

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), RuntimeError)
    assert outcome["elapsed"] < 5
