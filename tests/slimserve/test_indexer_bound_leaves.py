# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path

import pytest

from benchmarks.kernels import audit_glm53_indexer_correction_loader as audit
from benchmarks.kernels import audit_glm53_kv_loader as common_audit
from benchmarks.kernels import glm53_indexer_bound_leaves as leaves
from benchmarks.kernels import prepare_glm53_indexer_correction_loader as prep
from benchmarks.kernels.check_glm53_indexer_correction_loader import read_manifest
from slimserve.rmsnorm_diagnostic import sha
from tests.slimserve.test_kv_loader_audit import actual_receipts, prepared_fixture


def leaf_fixture(tmp_path, mode="correction"):
    qualified = dict(
        input_sha256=["i0", "i1"],
        phases=[
            dict(
                baseline_sha256=["kv", "q", f"old{p}"],
                candidate_sha256=["kv", "q", f"new{p}"],
                selected_sha256=f"flags{p}",
            )
            for p in range(2)
        ],
    )
    path = tmp_path / "qualified.json"
    path.write_text(json.dumps(qualified))
    entry = dict(
        path=str(path),
        sha256=sha(path),
        case=dict(rank=0, rows=1, seed=530901, magnitude=0.125),
    )
    binding = dict(graph="g.py", symbol="combo", source="combo.py")
    record = dict(
        status="complete",
        binding=binding,
        qualified_case=entry,
        input_sha256=qualified["input_sha256"],
        replay_guards_pass=True,
        observations=[
            dict(
                phase=p,
                outputs=qualified["phases"][p][
                    "candidate_sha256" if mode == "correction" else "baseline_sha256"
                ],
                **(
                    dict(selection_sha256=f"flags{p}", arena_guards_pass=True)
                    if mode == "correction"
                    else {}
                ),
            )
            for p in leaves.PHASE_ORDER
        ],
    )
    return record, binding, entry


@pytest.mark.parametrize("mode", ["control", "correction"])
def test_leaf_auditor_joins_each_phase_to_qualified_outputs(tmp_path, mode):
    record, binding, entry = leaf_fixture(tmp_path, mode)
    leaves.audit_leaf(record, binding, entry, mode)


@pytest.mark.parametrize(
    "change",
    [
        "phase",
        "missing",
        "output",
        "flags",
        "guard",
        "input",
        "binding",
        "failed",
        "replay",
    ],
)
def test_leaf_auditor_rejects_output_replay_or_guard_drift(tmp_path, change):
    record, binding, entry = leaf_fixture(tmp_path)
    record = copy.deepcopy(record)
    if change == "phase":
        record["observations"][0]["phase"] = 1
    elif change == "missing":
        record["observations"].pop()
    elif change == "output":
        record["observations"][2]["outputs"][2] = "bad"
    elif change == "flags":
        record["observations"][3]["selection_sha256"] = "bad"
    elif change == "guard":
        record["observations"][4]["arena_guards_pass"] = False
    elif change == "input":
        record["input_sha256"][0] = "bad"
    elif change == "binding":
        record["binding"]["symbol"] = "other"
    elif change == "failed":
        record["status"] = "failed"
    else:
        record["replay_guards_pass"] = False
    with pytest.raises(ValueError):
        leaves.audit_leaf(record, binding, entry, "correction")


def test_control_cannot_claim_a_selection_write(tmp_path):
    record, binding, entry = leaf_fixture(tmp_path, "control")
    record["observations"][0]["selection_sha256"] = "bad"
    with pytest.raises(ValueError, match="control unexpectedly"):
        leaves.audit_leaf(record, binding, entry, "control")


def test_actual_controller_receipt_policy_requires_distinct_correction_namespace(
    tmp_path, monkeypatch
):
    # Re-label real CPU-observed KV events as a fixture for the generic join.
    # This is not represented as a live correction-kernel qualification.
    manifest, _, graphs, binaries, events = actual_receipts(tmp_path, monkeypatch)
    manifest["mode"] = "correction"
    manifest["selection_capacity"] = 8192
    for target in manifest["targets"]["0"]:
        target["correction"] = target.pop("kv")
    for row in graphs["bindings"]:
        row["dispatch"] = audit.DISPATCH
        row["appended"].update(
            selection_capacity=8192,
            selection_bytes=8194 * 128,
            selection_dtype="uint8",
            selection_device="cuda:0",
        )
    for event in events:
        event["event"] = event["event"].replace("kv_", "indexer_correction_", 1)
        if "mode" in event:
            event["mode"] = "correction"
    events[0]["source_sha256"] = sha(audit.LOADER_SOURCE)
    result = common_audit.check_graph_records(
        manifest, events[0]["manifest_sha256"], graphs, binaries, events, workflow=audit
    )
    assert result["appended_launchers"] == 2
    graphs["bindings"][0]["appended"]["selection_bytes"] += 1
    with pytest.raises(ValueError, match="arena"):
        common_audit.check_graph_records(
            manifest,
            events[0]["manifest_sha256"],
            graphs,
            binaries,
            events,
            workflow=audit,
        )


def test_series_copies_private_sources_and_read_manifest_checks_order(
    tmp_path, monkeypatch
):
    base = prepared_fixture(tmp_path, monkeypatch)
    base.update(
        schema=prep.SCHEMA,
        qualification_sha256=prep.QUALIFICATION_SHA,
        selection_capacity=8192,
        qualified_leaf_cases=[],
    )
    from benchmarks.kernels.check_glm53_attention_overwrite import matrix

    base["qualified_leaf_cases"] = [
        dict(case=case, path="fixture", sha256="fixture") for case in matrix()
    ]
    for ts in base["targets"].values():
        ts[0]["correction"] = ts[0].pop("kv")
    monkeypatch.setattr(prep, "build_base", lambda: copy.deepcopy(base))
    output = tmp_path / "series"
    prep.prepare_series(output)
    for mode, rank in prep.ORDER:
        path = output / f"{mode}-rank{rank}/manifest.json"
        data, prepared = read_manifest(path)
        assert len(prepared["runs"]) == 8
        for name, digest in data["private_sources"].items():
            assert sha(Path(data["private_namespace"]) / name) == digest
    prepared["runs"].reverse()
    (output / "preparation.json").write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="prepared correction"):
        read_manifest(output / "control-rank0/manifest.json")
    with pytest.raises(ValueError, match="preserve"):
        prep.prepare_series(output)
