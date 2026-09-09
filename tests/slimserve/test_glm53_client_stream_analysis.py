# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from benchmarks.analyze_glm53_client_streams import analyze


def receipt():
    def row(start, chunks):
        return {
            "start": start,
            "first": start + chunks[0][0],
            "last": start + chunks[-1][0],
            "usage": {"completion_tokens": sum(n for _, n in chunks)},
            "chunks": [{"seconds": t, "tokens": n} for t, n in chunks],
        }

    return {
        "aggregate_output_tps": 2.0,
        "client_decode_tps": 2.5,
        "requests": [
            row(10.0, [(1, 1), (2, 1), (4, 2), (5, 1)]),
            row(10.5, [(2.5, 1), (3.5, 1), (4.5, 1), (5.5, 1)]),
        ],
    }


def test_staggered_prefill_window_keeps_all_gaps_and_original_metrics():
    data = receipt()
    original = copy.deepcopy(data)
    result = analyze(data)
    assert data == original
    assert result["recorded_e2e_tps"] == 2.0
    assert result["recorded_client_decode_tps"] == 2.5
    assert result["first_token_spread_s"] == 2.0
    overlap = result["all_active_intersection"]
    assert overlap["duration_s"] == 2.0
    assert overlap["observed_tokens"] == 5
    assert overlap["client_arrival_tps"] == 2.5
    assert result["requests"][0]["inter_chunk_ms"]["max"] == 2000.0
    assert result["requests"][0]["max_gap_after_chunk"] == 1


def test_no_all_active_intersection_does_not_fabricate_a_rate():
    data = receipt()
    for key in ("start", "first", "last"):
        data["requests"][1][key] += 10.0
    result = analyze(data)["all_active_intersection"]
    assert result["duration_s"] == 0
    assert result["observed_tokens"] == 0
    assert result["client_arrival_tps"] is None


@pytest.mark.parametrize("invalid", ["time", "count", "endpoint", "nan"])
def test_bad_chunk_receipts_are_rejected(invalid):
    data = receipt()
    row = data["requests"][0]
    if invalid == "time":
        row["chunks"][1]["seconds"] = 0
    elif invalid == "count":
        row["chunks"][1]["tokens"] = 3
    elif invalid == "endpoint":
        row["first"] += 0.2
    else:
        row["chunks"][1]["seconds"] = float("nan")
    with pytest.raises(ValueError):
        analyze(data)
