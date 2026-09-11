# SPDX-License-Identifier: Apache-2.0
"""Stopped, delayed and iteration-capped profiling must leave no annotation work."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler
from vllm.v1.worker import gpu_worker


class CountingProfiler(WorkerProfiler):
    def __init__(self, config):
        super().__init__(config)
        self.annotations = 0

    def _start(self):
        pass

    def _stop(self):
        pass

    def annotate_context_manager(self, name):
        self.annotations += 1
        return nullcontext()


def worker_for(profiler):
    return SimpleNamespace(
        profiler=profiler,
        vllm_config=SimpleNamespace(
            profiler_config=SimpleNamespace(detailed_trace_annotation=False)
        ),
    )


@pytest.mark.parametrize(
    "state", ["never_started", "stopped", "iteration_cap", "delayed"]
)
def test_inactive_profiler_does_not_build_annotations(monkeypatch, state):
    profiler = CountingProfiler(
        ProfilerConfig(
            max_iterations=1 if state == "iteration_cap" else 0,
            delay_iterations=2 if state == "delayed" else 0,
        )
    )
    if state != "never_started":
        profiler.start()
    if state == "stopped":
        profiler.stop()
    if state == "iteration_cap":
        profiler.step()  # annotate_profile's next step must stop the profiler.
    compute = Mock(side_effect=AssertionError("inactive annotation work"))
    monkeypatch.setattr(gpu_worker, "compute_iteration_details", compute)
    with gpu_worker.Worker.annotate_profile(worker_for(profiler), object()):
        pass
    compute.assert_not_called()
    assert profiler.annotations == 0


def test_delayed_start_still_steps_and_records_when_running(monkeypatch):
    profiler = CountingProfiler(ProfilerConfig(delay_iterations=2))
    profiler.start()
    compute = Mock(
        return_value=SimpleNamespace(
            num_ctx_requests=0,
            num_ctx_tokens=0,
            num_generation_requests=1,
            num_generation_tokens=1,
        )
    )
    monkeypatch.setattr(gpu_worker, "compute_iteration_details", compute)
    worker = worker_for(profiler)
    with gpu_worker.Worker.annotate_profile(worker, object()):
        pass
    assert profiler.annotations == 0
    with gpu_worker.Worker.annotate_profile(worker, object()):
        pass
    assert profiler.annotations == 1
    assert compute.call_count == 1
    profiler.stop()
    with gpu_worker.Worker.annotate_profile(worker, object()):
        pass
    assert profiler.annotations == 1
    assert compute.call_count == 1
