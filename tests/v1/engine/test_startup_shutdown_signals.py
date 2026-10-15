# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A stop request (SIGTERM/SIGINT) during engine startup must end the server.

Regression tests for a server that ignored SIGTERM while the model was
loading: the API server's startup handler raised KeyboardInterrupt, the
engine-handshake socket context swallowed it, and the API server went on to
wait up to VLLM_ENGINE_READY_TIMEOUT_S for an engine that had been stopped.
Multiproc workers whose parent died mid-load kept loading as orphans.

None of these tests need a GPU.
"""

import multiprocessing
import signal
import subprocess
import sys
import textwrap
import time

import pytest
import zmq

from vllm.utils.network_utils import zmq_socket_ctx
from vllm.v1.engine.utils import poll_engine_socket_or_exit

pytestmark = pytest.mark.cpu_test

# Generous next to the ~1 s these take when they work; far below the 60 s
# that the stand-in "model load" or READY wait would take when they don't.
EXIT_DEADLINE_S = 15.0


def _ipc_path(tmp_path, name: str) -> str:
    return f"ipc://{tmp_path}/{name}"


def _start(script: str, *args: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip() == "STARTED":
            return proc
    proc.wait()
    pytest.fail(f"child exited before startup (rc={proc.returncode})")


def _time_to_exit(proc: subprocess.Popen) -> float:
    start = time.monotonic()
    try:
        proc.wait(timeout=EXIT_DEADLINE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail(f"process still running {EXIT_DEADLINE_S}s after the signal")
    return time.monotonic() - start


def test_zmq_socket_ctx_propagates_keyboard_interrupt(tmp_path):
    with (
        pytest.raises(KeyboardInterrupt),
        zmq_socket_ctx(_ipc_path(tmp_path, "ki"), zmq.ROUTER, bind=True),
    ):
        raise KeyboardInterrupt("terminated")


# Mirrors the API server: run_server() turns SIGTERM/SIGINT into
# KeyboardInterrupt, launch_core_engines() waits for the engine handshake
# inside zmq_socket_ctx(), and MPClient then waits for the engine's READY.
_API_SERVER_STARTUP = """
    import signal, sys, time
    import zmq
    from vllm.utils.network_utils import zmq_socket_ctx

    def _interrupt_init(*_):
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_init)
    signal.signal(signal.SIGINT, _interrupt_init)

    with zmq_socket_ctx(sys.argv[1], zmq.ROUTER, bind=True) as handshake:
        print("STARTED", flush=True)
        handshake.poll(timeout=60_000)  # engine is loading the model
    time.sleep(60)  # the READY wait that follows a completed handshake
    print("STARTUP CONTINUED AFTER STOP REQUEST", flush=True)
"""


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_stop_signal_during_engine_handshake_exits(tmp_path, sig):
    proc = _start(_API_SERVER_STARTUP, _ipc_path(tmp_path, "handshake"))
    time.sleep(0.5)
    proc.send_signal(sig)
    _time_to_exit(proc)
    assert proc.returncode != 0
    assert proc.stdout is not None
    assert "STARTUP CONTINUED" not in proc.stdout.read()


def _mp_context():
    return multiprocessing.get_context("spawn")


def test_ready_wait_fails_fast_when_engine_exits(tmp_path):
    ctx = zmq.Context()
    try:
        sock = ctx.socket(zmq.PULL)
        sock.bind(_ipc_path(tmp_path, "ready"))
        engine = _mp_context().Process(target=time.sleep, args=(0.2,))
        engine.start()
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="Engine core initialization failed"):
            poll_engine_socket_or_exit(sock, [engine], timeout_ms=60_000)
        assert time.monotonic() - start < EXIT_DEADLINE_S
        engine.join()
    finally:
        ctx.destroy(linger=0)


def test_ready_wait_returns_message_or_timeout(tmp_path):
    ctx = zmq.Context()
    try:
        path = _ipc_path(tmp_path, "ready-ok")
        sock = ctx.socket(zmq.PULL)
        sock.bind(path)
        engine = _mp_context().Process(target=time.sleep, args=(30,))
        engine.start()
        try:
            assert not poll_engine_socket_or_exit(sock, [engine], timeout_ms=50)
            push = ctx.socket(zmq.PUSH)
            push.connect(path)
            push.send(b"READY")
            assert poll_engine_socket_or_exit(sock, [engine], timeout_ms=5_000)
            assert sock.recv() == b"READY"
        finally:
            engine.kill()
            engine.join()
    finally:
        ctx.destroy(linger=0)


# Mirrors worker_main(): the worker's SIGTERM handler raises SystemExit, and
# the death pipe is the only way it learns that its parent (the engine core)
# is gone. "ignore" stands in for a main thread stuck in a native call.
_WORKER_STARTUP = """
    import signal, sys, threading, time
    from multiprocessing import Pipe
    import vllm.v1.executor.multiproc_executor as mpe

    if sys.argv[1] == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        mpe._ORPHANED_WORKER_EXIT_GRACE_S = 0.5
    else:
        def _handler(*_):
            raise SystemExit(3)
        signal.signal(signal.SIGTERM, _handler)

    death_reader, death_writer = Pipe(duplex=False)
    mpe.WorkerProc.monitor_death_pipe(
        death_reader, threading.Event(), threading.Event(), []
    )
    print("STARTED", flush=True)
    death_writer.close()  # the parent exits while this worker loads weights
    time.sleep(60)
    print("WORKER OUTLIVED ITS PARENT", flush=True)
"""


@pytest.mark.parametrize("main_thread", ["interruptible", "ignore"])
def test_worker_exits_when_parent_dies_during_startup(main_thread):
    proc = _start(_WORKER_STARTUP, main_thread)
    _time_to_exit(proc)
    assert proc.returncode == (3 if main_thread == "interruptible" else 1)
    assert proc.stdout is not None
    assert "OUTLIVED" not in proc.stdout.read()


def test_worker_death_pipe_after_startup_only_stops_queues():
    """Once startup is done, parent exit keeps the existing behaviour: the
    worker's queues are shut down and the busy loop exits on its own."""
    import threading

    from vllm.v1.executor.multiproc_executor import WorkerProc

    class _Queue:
        def __init__(self):
            self.stopped = threading.Event()

        def shutdown(self):
            self.stopped.set()

    reader, writer = multiprocessing.Pipe(duplex=False)
    shutdown_requested, startup_done = threading.Event(), threading.Event()
    startup_done.set()
    queue = _Queue()
    WorkerProc.monitor_death_pipe(reader, shutdown_requested, startup_done, [queue])
    writer.close()
    assert queue.stopped.wait(5)
    assert shutdown_requested.is_set()
