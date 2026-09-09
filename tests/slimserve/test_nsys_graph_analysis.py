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


def test_missing_graph_activity_retains_runtime_launch_and_warning(tmp_path):
    path = tmp_path / "missing-activity.sqlite"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE StringIds (id, value)")
        database.execute("INSERT INTO StringIds VALUES (1, 'cudaGraphLaunch_v10000')")
        database.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "(start, end, globalTid, correlationId, nameId)"
        )
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,2,3,4,1)")
        database.execute("CREATE TABLE DIAGNOSTIC_EVENT (text)")
        database.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES ('Missing CUDA events')")
    result = analyze(path)
    assert result["status"] == "incomplete"
    assert result["missing_graph_activity_table"]
    assert result["graph_record_count"] == 0
    assert result["runtime_graph_launch_count"] == 1
    assert result["runtime_graph_launches"][0]["correlationId"] == 4
    assert result["collection_diagnostics"] == [{"text": "Missing CUDA events"}]
