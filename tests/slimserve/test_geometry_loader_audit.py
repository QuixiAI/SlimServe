# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from torch._inductor.codecache import PyCodeCache

from benchmarks.kernels.audit_glm53_geometry_graphs import inventory
from benchmarks.kernels.audit_glm53_geometry_loader import check_records, json_lines
from benchmarks.kernels.check_glm53_geometry_loader import (
    ORDER,
    checked_bundles,
    environment,
    prior_receipts,
)
from slimserve.rmsnorm_diagnostic import sha
from tests.slimserve.test_geometry_loader import fixture


def actual_receipts(tmp_path, monkeypatch, mode="geometry"):
    loader, manifest, graph, _, _ = fixture(tmp_path, monkeypatch, mode)
    binaries = []
    loader.observer.emit = binaries.append
    with loader.intercept():
        PyCodeCache.load_by_key_path("auditor-fixture", str(graph))
        report = inventory(PyCodeCache.modules, manifest, 0, mode, loader.observer)
        loader.controller.seal(PyCodeCache.modules)
        loader.observer.seal()
        loader.controller.verify_graphs(PyCodeCache.modules)
    manifest.update(rank=0, mode=mode)
    digest = sha(loader.manifest_path)
    summary = dict(
        status="complete",
        rank=0,
        mode=mode,
        manifest_sha256=digest,
        model_forward_calls=0,
        weight_tensors_loaded=0,
        artifacts=7,
        loaded_artifacts=7,
        submodules=46,
        original_cache_unchanged=True,
        static_bundles=[dict(expected=4, loaded=4)] * 7,
        observed_binary_objects=len(loader.observer.records),
    )
    events = [
        json_lines(child.path) for child in loader.controller.controllers.values()
    ]
    loader.close()
    return manifest, digest, summary, report, binaries, events


@pytest.mark.parametrize("mode", ["control", "geometry"])
def test_offline_join_accepts_actual_cpu_loader_receipts(tmp_path, monkeypatch, mode):
    records = actual_receipts(tmp_path, monkeypatch, mode)
    result = check_records(*records)
    assert result["target_bindings"] == 3 and result["bound_launchers"] == 4


@pytest.mark.parametrize(
    "change",
    [
        "failed",
        "partial_artifact",
        "fallback",
        "model_forward",
        "weights",
        "unsealed",
        "late_binary",
        "binary_key",
        "binary_bytes",
        "handles",
        "graph_missing",
        "wrong_selection",
        "wrong_target",
        "unobserved",
        "duplicate_binding",
        "missing_stream",
        "controller_unsealed",
        "controller_before",
        "controller_after",
        "controller_index",
        "controller_coverage",
        "controller_graph",
        "manifest",
    ],
)
def test_offline_join_rejects_changed_receipts(tmp_path, monkeypatch, change):
    manifest, digest, summary, graphs, binaries, events = actual_receipts(
        tmp_path, monkeypatch
    )
    if change == "failed":
        summary["status"] = "failed"
    elif change == "partial_artifact":
        summary["loaded_artifacts"] = 6
    elif change == "fallback":
        summary["static_bundles"][0]["loaded"] = 3
    elif change == "model_forward":
        summary["model_forward_calls"] = 1
    elif change == "weights":
        summary["weight_tensors_loaded"] = 1
    elif change == "unsealed":
        binaries.pop()
    elif change == "late_binary":
        binaries.insert(0, binaries.pop())
    elif change == "binary_key":
        binaries[0]["metadata"]["hash"] = "00" * 32
    elif change == "binary_bytes":
        Path(binaries[0]["path"]).write_bytes(b"drift")
    elif change == "handles":
        binaries[0]["handles"][0] = 0
    elif change == "graph_missing":
        graphs["bindings"].pop(0)
    elif change == "wrong_selection":
        graphs["bindings"][0]["selected"][0]["config"]["R0_BLOCK"] = 32
    elif change == "wrong_target":
        graphs["bindings"][0]["target"] = False
    elif change == "unobserved":
        graphs["bindings"][0]["observed_binary_index"] = 999
    elif change == "duplicate_binding":
        graphs["bindings"].append(copy.deepcopy(graphs["bindings"][0]))
    elif change == "missing_stream":
        events.pop()
    elif change == "controller_unsealed":
        events[0] = [r for r in events[0] if r["event"] != "sealed"]
    elif change == "controller_before":
        next(r for r in events[0] if r["event"] == "launcher")["before"] = []
    elif change == "controller_after":
        next(r for r in events[0] if r["event"] == "launcher")["after"] = []
    elif change == "controller_index":
        next(r for r in events[0] if r["event"] == "launcher")["binding_index"] = 12
    elif change == "controller_coverage":
        next(r for r in events[0] if r["event"] == "graph_coverage")["bindings"] = 12
    elif change == "controller_graph":
        events[0] = [r for r in events[0] if r["event"] != "graph_binding"]
    else:
        digest = "wrong"
    with pytest.raises((ValueError, KeyError)):
        check_records(manifest, digest, summary, graphs, binaries, events)


def test_bundle_hook_checks_fallback_and_restores_exact_descriptor():
    class Bundle:
        @classmethod
        def load_autotuners(cls, tuners):
            return tuners[:-1]

    original = Bundle.__dict__["load_autotuners"]
    counts = []
    with pytest.raises(ValueError, match="fallback"), checked_bundles(Bundle, counts):
        Bundle.load_autotuners([1, 2])
    assert counts == [dict(expected=2, loaded=1)]
    assert Bundle.__dict__["load_autotuners"] is original


def test_loader_environment_preserves_rank_cache_semantics():
    manifest = dict(cache_root="/private/cache", private_namespace="/private/cache/ns")
    with patch.dict(os.environ, {"TRITON_CACHE_DIR": "/wrong/shared"}, clear=True):
        environment(manifest)
        assert "TRITON_CACHE_DIR" not in os.environ
        assert (
            os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/private/cache/ns/inductor_cache"
        )
        assert os.environ["SLIMSERVE_GLM53_NATIVE_ORDER"] == "1"


@pytest.mark.parametrize(
    "flag",
    [
        "SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC",
        "TORCHINDUCTOR_DETERMINISTIC",
        "VLLM_BATCH_INVARIANT",
    ],
)
def test_loader_environment_refuses_other_diagnostics(flag):
    with patch.dict(os.environ, {flag: "1"}, clear=True), pytest.raises(ValueError):
        environment({})


def test_prescribed_predecessors_must_all_have_unchanged_successful_audits(tmp_path):
    rows = []
    for mode, rank in ORDER:
        folder = tmp_path / f"{mode}-rank{rank}"
        (folder / "run").mkdir(parents=True)
        summary = folder / "run/summary.json"
        summary.write_text("{}\n")
        launch_record = dict(
            status="complete", manifest_sha256=f"{mode}-{rank}", gpu_processes_after=""
        )
        for kind in ("load", "audit"):
            log = folder / f"{kind}.log"
            log.write_text("done\n")
            launch_record[kind] = dict(exit_code=0, log_sha256=sha(log))
        (folder / "launch.json").write_text(json.dumps(launch_record))
        (folder / "run/analysis.json").write_text(
            json.dumps(
                dict(
                    status="complete",
                    manifest_sha256=f"{mode}-{rank}",
                    summary_sha256=sha(summary),
                    receipts={str(summary): sha(summary)},
                )
            )
        )
        rows.append(
            dict(
                mode=mode,
                rank=rank,
                manifest=str(folder / "manifest.json"),
                manifest_sha256=f"{mode}-{rank}",
            )
        )
    manifest = dict(mode="geometry", rank=3)
    preparation = dict(runs=rows)
    assert len(prior_receipts(None, manifest, preparation)) == 7
    earliest = tmp_path / "control-rank0/run/summary.json"
    earliest.write_text("drift\n")
    with pytest.raises(ValueError, match="predecessors"):
        prior_receipts(None, manifest, preparation)
