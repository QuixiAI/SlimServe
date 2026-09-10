# SPDX-License-Identifier: Apache-2.0
from benchmarks.summarize_glm5_next_telemetry import summarize_samples


def cycle(t, utilizations):
    return [
        {
            "timestamp": t + i * 0.01,
            "index": i,
            "utilization": u,
            "sm_clock": 1410,
            "power": 200,
        }
        for i, u in enumerate(utilizations)
    ]


def test_empty():
    assert summarize_samples([])["samples"] == 0


def test_all_devices_must_be_present_and_idle():
    rows = cycle(0, [0, 0]) + cycle(2, [0, 1]) + cycle(4, [0])
    spans = summarize_samples(rows, 2)["all_gpu_idle_observed_spans"]
    assert len(spans) == 1
    assert spans[0]["observed_span_seconds"] == 0


def test_contiguous_idle_and_telemetry_gaps():
    rows = cycle(0, [0, 0]) + cycle(2, [0, 0]) + cycle(20, [0, 0])
    spans = summarize_samples(rows, 2)["all_gpu_idle_observed_spans"]
    assert [s["observed_span_seconds"] for s in spans] == [2, 0]


def test_activity_breaks_an_idle_span():
    rows = cycle(0, [0, 0]) + cycle(2, [10, 0]) + cycle(4, [0, 0])
    assert len(summarize_samples(rows, 2)["all_gpu_idle_observed_spans"]) == 2


def test_partial_first_cycle_is_not_all_gpu_idle():
    rows = cycle(0, [0, 0])[1:] + cycle(2, [0, 0])
    summary = summarize_samples(rows, 2)
    assert summary["samples"] == 3
    assert summary["all_gpu_idle_observed_spans"][0]["first_sample_utc"] == 2
