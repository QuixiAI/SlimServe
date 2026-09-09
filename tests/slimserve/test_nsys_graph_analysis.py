# SPDX-License-Identifier: Apache-2.0
import sqlite3

import pytest

from benchmarks.analyze_nsys_graph_trace import analyze


def test_graph_spans_keep_rank_identity_and_first_replay(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as database:
        database.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "(start, end, deviceId, globalPid, contextId, streamId, "
            "graphId, graphExecId, correlationId)"
        )
        database.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (0, 3000, 0, 100, 1, 2, 7, 9, 10),
                (4000, 5000, 0, 100, 1, 2, 7, 9, 11),
                (0, 2000, 1, 200, 1, 2, 7, 9, 10),
                (7000, 0, 1, 200, 1, 2, 7, 9, 11),
            ],
        )
    result = analyze(path)
    assert result["status"] == "incomplete"
    assert result["graph_record_count"] == 4
    assert len(result["invalid_boundary_records"]) == 1
    assert len(result["groups"]) == 2
    assert result["groups"][0]["count"] == 2
    assert result["groups"][0]["duration_us"]["median"] == 2.0
    assert result["groups"][0]["duration_us"]["max"] == 3.0


def test_rejects_non_graph_capture_without_creating_missing_input(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        analyze(path)
    assert not path.exists()
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE other (value)")
    with pytest.raises(ValueError, match="no whole-graph"):
        analyze(path)
