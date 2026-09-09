# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.analyze_glm53_rmsnorm_graphs import audit_rank, expected


@pytest.fixture
def fixture():
    configs = [
        dict(triton_cache_hash=h, XBLOCK=1, R0_BLOCK=r, num_warps=w, num_stages=1)
        for h, r, w in (("legacy", 1024, 8), ("native", 4096, 16))
    ]
    native, selected = map(expected, configs[::-1])
    target = dict(filename="norm.py", source_sha256="norm-sha", configs=configs)
    manifest = dict(
        private_namespace="/private",
        targets={"0": target},
        original_files={"norm.py": "norm-sha", "graph.py": "graph-sha"},
    )
    begin = dict(
        event="begin",
        rank=0,
        mode="legacy",
        manifest_sha256="manifest",
        source_sha256="diagnostic",
    )
    launcher = dict(
        event="launcher",
        rank=0,
        filename="/private/norm.py",
        target=True,
        before=native,
        after=selected,
        resolved_by="graph",
        sealed=False,
        resolution_index=1,
        binding_index=1,
        repeated=False,
    )
    graph = dict(
        event="graph_binding",
        rank=0,
        filename="/private/norm.py",
        module="/private/graph.py",
        module_sha256="graph-sha",
        symbol="norm",
        binding_index=1,
        selected=selected,
    )
    records = [
        begin,
        launcher,
        graph,
        dict(event="graph_coverage", rank=0, bindings=1),
        dict(event="sealed", rank=0, sources=1, targets=1, resolutions=1),
    ]
    kwargs = dict(
        rank=0,
        arm="legacy",
        manifest=manifest,
        manifest_sha256="manifest",
        source_sha256="diagnostic",
        expected_graphs={("graph.py", "norm"): "graph-sha"},
        hash_file=lambda path: manifest["original_files"][
            str(path.relative_to(Path("/private")))
        ],
    )
    return records, kwargs


def test_graph_only_resolution(fixture):
    records, kwargs = fixture
    result = audit_rank(records, **kwargs)
    assert result["distinct_target_objects"] == 1
    assert len(result["graph_bindings"]) == 1


def test_repeated_graph_alias_and_post_seal_reuse(fixture):
    records, kwargs = fixture
    alias = deepcopy(records[1])
    alias.update(resolution_index=2, repeated=True, before=alias["after"])
    records[3:3] = [alias, deepcopy(records[2])]
    records[-1]["resolutions"] = 2
    post = {**alias, "resolution_index": 3, "sealed": True, "resolved_by": "reuse"}
    records.append(post)
    result = audit_rank(records, **kwargs)
    assert len(result["targets"]) == 3
    assert len(result["graph_bindings"]) == 1


def test_multiple_objects_for_one_source(fixture):
    records, kwargs = fixture
    initial = {**deepcopy(records[1]), "resolved_by": "upstream"}
    records.insert(1, initial)
    records[2].update(resolution_index=2, binding_index=2)
    records[3]["binding_index"] = 2
    records[-1].update(targets=2, resolutions=2)
    result = audit_rank(records, **kwargs)
    assert result["distinct_target_objects"] == 2


@pytest.mark.parametrize(
    "change",
    [
        "missing_graph",
        "static_only",
        "graph_binary",
        "graph_config",
        "graph_sha",
        "symbol",
        "coverage_count",
        "source",
        "outside",
        "rank",
        "manifest",
        "diagnostic",
        "binding",
        "resolution",
        "late_binding",
        "before",
        "selected",
        "seal",
        "unknown_event",
        "graph_without_resolution",
    ],
)
def test_corruption_fails_closed(fixture, change):
    rows, kwargs = fixture
    if change == "missing_graph":
        del rows[2]
    elif change == "static_only":
        del rows[2:4]
    elif change == "graph_binary":
        rows[2]["selected"] = [{"hash": "wrong"}]
    elif change == "graph_config":
        rows[2]["selected"] = deepcopy(rows[2]["selected"])
        rows[2]["selected"][0]["config"]["num_warps"] = 32
    elif change == "graph_sha":
        rows[2]["module_sha256"] = "wrong"
    elif change == "symbol":
        rows[2]["symbol"] = "renamed"
    elif change == "coverage_count":
        rows[3]["bindings"] = 2
    elif change == "source":
        kwargs["hash_file"] = lambda _: "wrong"
    elif change == "outside":
        rows[1]["filename"] = "/original/norm.py"
    elif change == "rank":
        rows[2]["rank"] = 1
    elif change == "manifest":
        rows[0]["manifest_sha256"] = "wrong"
    elif change == "diagnostic":
        rows[0]["source_sha256"] = "wrong"
    elif change == "binding":
        rows[2]["binding_index"] = 2
    elif change == "resolution":
        rows[1]["resolution_index"] = 2
    elif change == "late_binding":
        rows[1]["sealed"] = True
    elif change == "before":
        rows[1]["before"] = rows[1]["after"]
    elif change == "selected":
        rows[1]["after"] = rows[1]["before"]
    elif change == "seal":
        rows[-1]["targets"] = 2
    elif change == "unknown_event":
        rows[2]["event"] = "mystery"
    else:
        rows[1]["resolved_by"] = "upstream"
    with pytest.raises(ValueError):
        audit_rank(rows, **kwargs)


def test_partial_graph_inventory_rejected_even_if_count_agrees(fixture):
    rows, kwargs = fixture
    kwargs["expected_graphs"]["second.py", "norm"] = "second-sha"
    with pytest.raises(ValueError, match="incomplete"):
        audit_rank(rows, **kwargs)


def test_noop_control(fixture):
    rows, kwargs = fixture
    kwargs["arm"] = rows[0]["mode"] = "control"
    rows[1]["after"] = rows[1]["before"]
    rows[2]["selected"] = rows[1]["before"]
    audit_rank(rows, **kwargs)
