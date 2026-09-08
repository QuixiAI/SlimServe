# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "benchmarks" / "analyze_cuda_graph_trace.py"
spec = importlib.util.spec_from_file_location("graph_trace", SCRIPT)
graph_trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_trace)


def kernel(start, duration, correlation, stream=1):
    return {
        "cat": "kernel",
        "name": "example",
        "ts": start,
        "dur": duration,
        "args": {
            "device": 0,
            "graph id": 7,
            "correlation": correlation,
            "stream": stream,
        },
    }


def test_busy_union_does_not_double_count_overlap():
    assert (
        graph_trace.busy_union(
            [kernel(0, 10, 1), kernel(3, 2, 1), kernel(8, 4, 1), kernel(15, 2, 1)]
        )
        == 14
    )


def test_ahead_of_gpu_host_launches_do_not_define_replay_windows():
    events = [
        {"name": "cudaGraphLaunch", "ts": 0, "args": {"correlation": 1}},
        {"name": "cudaGraphLaunch", "ts": 1, "args": {"correlation": 2}},
        kernel(100, 10, 1),
        kernel(105, 20, 1, stream=2),
        kernel(127, 3, 1),
        kernel(150, 10, 2),
    ]
    result = graph_trace.analyze(events)
    first, second = result["replays"]
    assert first["span_us"] == 30
    assert first["kernel_duration_sum_us"] == 33
    assert first["kernel_busy_union_us"] == 28
    assert first["no_kernel_active_us"] == 2
    assert first["gpu_start_after_host_launch_us"] == 100
    assert second["gpu_start_after_host_launch_us"] == 149
    # A partially observed replay is retained, not silently filtered away.
    assert result["summaries"][0]["observed_kernel_counts"] == {3: 1, 1: 1}
