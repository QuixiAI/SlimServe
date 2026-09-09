# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.analyze_model_step_trace import analyze


def step(start, duration, name="execute_prefill"):
    return dict(
        cat="user_annotation", name=name, ph="X", pid=1, tid=2, ts=start, dur=duration
    )


def launch(time, correlation, category="cuda_runtime", tid=2):
    return dict(
        cat=category,
        name="launch",
        pid=1,
        tid=tid,
        ts=time,
        dur=1,
        args={"correlation": correlation},
    )


def gpu(start, duration, correlation, category="kernel"):
    return dict(
        cat=category,
        name="op",
        ts=start,
        dur=duration,
        args={"correlation": correlation, "device": 0},
    )


def test_queued_gpu_work_belongs_to_launch_step_not_gpu_timestamp_window():
    result = analyze(
        [
            step(0, 10, "execute_first"),
            step(20, 10, "execute_second"),
            launch(5, 1),
            launch(25, 2, "cuda_driver"),
            gpu(22, 5, 1),
            gpu(40, 8, 2),
        ]
    )
    assert result["steps"][0]["devices"][0]["gpu_start_us"] == 22
    assert result["steps"][1]["devices"][0]["gpu_start_us"] == 40
    assert result["unmatched_gpu_count"] == 0


def test_overlap_union_orphans_and_empty_steps_remain_visible():
    result = analyze(
        [
            step(0, 10),
            step(20, 5),
            launch(2, 1),
            launch(3, 2),
            launch(4, 3, tid=9),
            gpu(12, 10, 1),
            gpu(17, 10, 2, "gpu_memcpy"),
            gpu(30, 3, 3),
            gpu(40, 1, 99),
        ]
    )
    row = result["steps"][0]["devices"][0]
    assert row["gpu_span_us"] == row["gpu_busy_union_us"] == 15
    assert row["kernel_busy_union_us"] == 10
    assert row["no_gpu_activity_us"] == 0
    assert row["kernel_count"] == 1
    assert result["steps"][1]["devices"] == []
    assert result["unmatched_gpu_count"] == 2


def test_ambiguous_host_annotations_fail_instead_of_selecting_one():
    with pytest.raises(ValueError, match="overlapping execute"):
        analyze([step(0, 10), step(1, 5), launch(3, 1)])
