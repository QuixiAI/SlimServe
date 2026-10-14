# SPDX-License-Identifier: Apache-2.0
"""A graph's internal streams must not be mistaken for missing model layers."""

import pytest

from benchmarks.summarize_glm5_next_trace import summarize_events


def event(name, start, duration, stream, cat="kernel"):
    return dict(
        name=name,
        ts=start,
        dur=duration,
        pid=0,
        tid=stream,
        cat=cat,
        args={"stream": stream},
    )


def fixture_events(other_stream):
    return [
        event("execute_1_context_0_generation_1", 0, 100, 23, "gpu_user_annotation"),
        event("_mhc_partials", 10, 10, 23),
        event("attention", 15, 15, 23),
        event("_mhc_partials", 40, 10, other_stream),
    ]


CONFIG = {"mlp_layer_types": ["dense"], "linear_attn_config": {"full_attn_layers": [0]}}


def test_multistream_counts_every_marker_without_inventing_site_order():
    result = summarize_events(fixture_events(306), CONFIG)
    assert result["total_mhc_markers"] == 2
    assert result["sites"] == {}
    assert result["site_attribution"].startswith("unavailable")
    assert result["kernel_busy_union_us"] == 30
    assert result["kernel_sum_us"] == 35
    assert result["region_us"] == 100
    assert result["all_stream_kernels"][0]["calls"] == 2


def test_single_stream_preserves_site_attribution():
    result = summarize_events(fixture_events(23), CONFIG)
    assert result["site_attribution"] == "single_stream"
    assert set(result["sites"]) == {"MLA", "dense_MLP"}


def test_truncated_region_still_fails_completeness_gate():
    with pytest.raises(ValueError, match="Expected 2 mHC sites, got 1"):
        summarize_events(fixture_events(306)[:-1], CONFIG)


def test_requested_batch_size_does_not_select_other_decode_regions():
    events = fixture_events(306)
    for item in events:
        item["ts"] += 200
    batch = fixture_events(306)
    batch[0]["name"] = "execute_32_context_0_generation_32"
    result = summarize_events(batch + events, CONFIG, tokens=32)
    assert result["tokens"] == 32
    assert result["region"] == batch[0]["name"]
    assert result["total_mhc_markers"] == 2
    with pytest.raises(ValueError, match="No 16-token"):
        summarize_events(batch + events, CONFIG, tokens=16)


def test_nonpositive_token_count_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        summarize_events(fixture_events(306), CONFIG, tokens=0)
