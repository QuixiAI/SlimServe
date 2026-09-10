# SPDX-License-Identifier: Apache-2.0
import json
import os
from types import SimpleNamespace

import pytest

from benchmarks.kernels import run_glm53_geometry_serving as runner
from benchmarks.kernels.glm53_geometry_workload import FLAG, MANIFEST, command
from slimserve.rmsnorm_diagnostic import sha


def fixture(tmp_path, label="control"):
    folder = tmp_path / label
    folder.mkdir()
    path = folder / "manifest.json"
    manifest = dict(
        label=label,
        mode="geometry" if label == "geometry" else "control",
        cache_root=str(folder / "cache"),
        private_namespace=str(folder / "cache/private"),
        sources={},
        original_files={},
        workload=dict(
            prompt="prompt.txt",
            environment={
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "SLIMSERVE_CACHE": "/raid/weights",
                "CUDA_HOME": "/usr/local/cuda-13.0",
                "OMP_NUM_THREADS": "1",
                "SLIMSERVE_GLM53_NATIVE_ORDER": "1",
                "VLLM_GLM5_MHC_BF16_FN": "1",
                "VLLM_GLM5_MHC_PREFILL_TC": "0",
            },
        ),
    )
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_environment_preserves_fixed_recipe_and_clears_shell_overrides(
    tmp_path, monkeypatch
):
    path, manifest = fixture(tmp_path)
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/unrelated")
    monkeypatch.setenv("TORCHINDUCTOR_DETERMINISTIC", "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "0")
    for cpu in (False, True):
        env = runner.environment(path, manifest, cpu=cpu)
        assert env["CUDA_VISIBLE_DEVICES"] == ("" if cpu else "0,1,2,3")
        assert env["SLIMSERVE_CACHE"] == "/raid/weights"
        assert env["VLLM_FORCE_AOT_LOAD"] == "1"
        assert env[FLAG] == "control" and env[MANIFEST] == str(path)
        assert all(
            k not in env
            for k in (
                "NCCL_P2P_DISABLE",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_DETERMINISTIC",
            )
        )
    assert os.environ["NCCL_P2P_DISABLE"] == "1"
    before = dict(os.environ)
    with pytest.raises(ValueError), runner.local_environment({"ONLY": "value"}):
        assert dict(os.environ) == {"ONLY": "value"}
        raise ValueError("test")
    assert dict(os.environ) == before


@pytest.mark.parametrize("kind,memory,cpu", [("serve", 150, False), ("audit", 8, True)])
def test_execute_bounded_scope_and_exact_command(
    tmp_path, monkeypatch, kind, memory, cpu
):
    path, manifest = fixture(tmp_path)
    record = {}

    def execute(argv, **kwargs):
        assert f"MemoryMax={memory}G" in argv and "MemorySwapMax=0" in argv
        assert kwargs["cwd"] == runner.ROOT
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ("" if cpu else "0,1,2,3")
        if not cpu:
            assert argv[9:] == command(path, manifest)
        kwargs["stdout"].write("preserved process output\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.execute(kind, path, manifest, record) == 0
    assert record[kind]["log_sha256"] == sha(path.parent / f"{kind}.log")


@pytest.mark.parametrize(
    "failed", [None, "preflight", "serve", "audit", "release", "query"]
)
def test_launch_retains_failures_audits_partial_work_and_never_retries(
    tmp_path, monkeypatch, failed
):
    path, manifest = fixture(tmp_path)
    events = []
    monkeypatch.setattr(runner, "frozen_manifest", lambda p: manifest)

    def preflight(*args):
        events.append("preflight")
        if failed == "preflight":
            raise ValueError("failed preflight")
        return {"priors": []}

    def execute(kind, path, manifest, record):
        events.append(kind)
        log = path.parent / f"{kind}.log"
        log.write_text(f"{kind} attempt\n")
        record[kind] = dict(exit_code=int(failed == kind), log_sha256=sha(log))
        return record[kind]["exit_code"]

    monkeypatch.setattr(runner, "preflight", preflight)
    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "verify", lambda m: None)

    def query():
        if failed == "query":
            raise OSError("query failed")
        return "foreign GPU" if failed == "release" else ""

    monkeypatch.setattr(runner, "gpu_processes", query)
    if failed:
        with pytest.raises(ValueError):
            runner.launch(path)
    else:
        runner.launch(path)
    record = runner.read(path.parent / "launch.json")
    assert record["status"] == ("failed" if failed else "complete")
    assert record["files"] == runner.inventory(path.parent)
    assert events == (
        ["preflight"] if failed == "preflight" else ["preflight", "serve", "audit"]
    )
    original = sha(path.parent / "launch.json")
    with pytest.raises((ValueError, FileExistsError)):
        runner.launch(path)
    assert sha(path.parent / "launch.json") == original


@pytest.mark.parametrize(
    "label,expected",
    [
        ("control", []),
        ("geometry", ["control"]),
        ("return-control", ["control", "geometry"]),
    ],
)
def test_preflight_checks_every_prescribed_predecessor(
    tmp_path, monkeypatch, label, expected
):
    path, manifest = fixture(tmp_path, label)
    seen = []

    def check(previous):
        seen.append(previous.parent.name)
        return {"label": previous.parent.name}

    monkeypatch.setattr(runner, "check_completed", check)
    monkeypatch.setattr(runner, "inventory", lambda p: {"unexpected": "hash"})
    with pytest.raises(ValueError, match="fresh private cache"):
        runner.preflight(path, manifest)
    assert seen == expected


def test_preflight_stops_before_other_work_when_predecessor_failed(
    tmp_path, monkeypatch
):
    path, manifest = fixture(tmp_path, "return-control")
    seen = []

    def check(previous):
        seen.append(previous.parent.name)
        raise ValueError("failed predecessor")

    monkeypatch.setattr(runner, "check_completed", check)
    monkeypatch.setattr(
        runner, "gpu_processes", lambda: pytest.fail("must not query or launch")
    )
    with pytest.raises(ValueError, match="failed predecessor"):
        runner.preflight(path, manifest)
    assert seen == ["control"]


def test_inventory_covers_cache_and_evidence_but_not_self_referential_marker(tmp_path):
    (tmp_path / "launch.json").write_text("marker")
    (tmp_path / "cache").mkdir()
    path = tmp_path / "cache/kernel.cubin"
    path.write_bytes(b"binary")
    assert runner.inventory(tmp_path) == {"cache/kernel.cubin": sha(path)}
    (tmp_path / "alias").symlink_to(path)
    with pytest.raises(ValueError, match="alias"):
        runner.inventory(tmp_path)


@pytest.mark.parametrize("changed", [None, "log", "cache", "quality", "release"])
def test_completed_predecessor_binds_all_preserved_receipts(tmp_path, changed):
    path, manifest = fixture(tmp_path)
    folder = path.parent
    audit = dict(
        status="complete",
        diagnostic_gate_passed=True,
        manifest_sha256=sha(path),
        production_qualified=False,
        quality_passed=False,
    )
    (folder / "workload-analysis.json").write_text(json.dumps(audit))
    (folder / "worker-analysis.json").write_text(json.dumps(audit))
    record = dict(
        status="complete",
        label="control",
        manifest_sha256=sha(path),
        gpu_processes_after="",
        gpu_processes_after_audit="",
    )
    for kind in ("serve", "audit"):
        (folder / f"{kind}.log").write_text(kind)
        record[kind] = dict(exit_code=0, log_sha256=sha(folder / f"{kind}.log"))
    record["files"] = runner.inventory(folder)
    if changed == "release":
        record["gpu_processes_after"] = "active"
    (folder / "launch.json").write_text(json.dumps(record))
    if changed == "log":
        (folder / "serve.log").write_text("changed")
    elif changed == "cache":
        (folder / "extra.cubin").write_bytes(b"extra")
    elif changed == "quality":
        (folder / "workload-analysis.json").write_text("{}")
    if changed:
        with pytest.raises(ValueError):
            runner.check_completed(path)
    else:
        assert runner.check_completed(path)["label"] == "control"


def test_gpu_query_failure_is_not_release(monkeypatch):
    def query(*args, **kwargs):
        assert kwargs["timeout"] == 10
        raise OSError("driver query failed")

    monkeypatch.setattr(runner.subprocess, "check_output", query)
    with pytest.raises(OSError):
        runner.gpu_processes()


def test_failed_series_closure_keeps_unused_cases_terminal(tmp_path, monkeypatch):
    manifests = {}
    for label, _ in runner.CASES:
        path, manifest = fixture(tmp_path, label)
        manifests[path] = manifest
        if label == "control":
            record = dict(status="failed", files=runner.inventory(path.parent))
            (path.parent / "launch.json").write_text(json.dumps(record))
    monkeypatch.setattr(runner, "frozen_manifest", lambda p: manifests[p])
    monkeypatch.setattr(runner, "gpu_processes", lambda: "")
    assert runner.close_series(tmp_path)
    closure = runner.read(tmp_path / "closure.json")
    assert closure["status"] == "terminal-failure"
    assert [r["status"] for r in closure["cases"]] == [
        "failed",
        "unlaunched",
        "unlaunched",
    ]
    with pytest.raises(ValueError, match="preserve prior"):
        runner.close_series(tmp_path)


def test_interrupted_execute_stops_only_its_named_scope_and_preserves_log(
    tmp_path, monkeypatch
):
    path, manifest = fixture(tmp_path)
    calls, record = [], {}

    def execute(argv, **kwargs):
        calls.append(argv)
        kwargs["stdout"].write("retained output\n")
        if len(calls) == 1:
            raise KeyboardInterrupt()
        assert argv == [
            "systemctl",
            "--user",
            "stop",
            calls[0][3].removeprefix("--unit=") + ".scope",
        ]
        assert kwargs["timeout"] == 30
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    with pytest.raises(KeyboardInterrupt):
        runner.execute("serve", path, manifest, record)
    assert len(calls) == 2 and record["serve"]["status"] == "failed"
    assert record["serve"]["log_sha256"] == sha(path.parent / "serve.log")


@pytest.mark.parametrize("mutation", [None, "non-target", "return-target", "prefill"])
def test_complete_closure_compares_actual_kernel_bindings(
    tmp_path, monkeypatch, mutation
):
    manifests = {}
    for label, _ in runner.CASES:
        path, manifest = fixture(tmp_path, label)
        manifests[path] = manifest
        audit = dict(
            diagnostic_tps={},
            prefill_prompt_sha256={"32768": "prompt", "131072": "long"},
            quality_passed=label != "geometry",
            historical_comparisons={
                "control": {"exact": label != "geometry"},
                "failed-no-combo": {"exact": label == "geometry"},
            },
        )
        if label == "geometry" and mutation == "prefill":
            audit["prefill_prompt_sha256"]["131072"] = "wrong"
        (path.parent / "workload-analysis.json").write_text(json.dumps(audit))
        for rank in range(4):
            folder = path.parent / f"worker-receipts/rank-{rank}"
            folder.mkdir(parents=True)
            target = dict(
                graph="graph",
                symbol="target",
                source="target.py",
                target=True,
                config=label == "geometry",
            )
            other = dict(
                graph="graph", symbol="other", source="other.py", target=False, config=1
            )
            if label == "geometry" and mutation == "non-target":
                other["config"] = 2
            if label == "return-control" and mutation == "return-target":
                target["config"] = "wrong"
            (folder / "capture-after-bindings.json").write_text(
                json.dumps(dict(bindings=[target, other]))
            )
    monkeypatch.setattr(runner, "frozen_manifest", lambda p: manifests[p])
    monkeypatch.setattr(runner, "gpu_processes", lambda: "")
    monkeypatch.setattr(runner, "check_completed", lambda p: {"label": p.parent.name})
    assert runner.close_series(tmp_path) is (mutation is None)
    result = runner.read(tmp_path / "closure.json")
    assert not result["production_qualified"]
    if mutation is None:
        assert result["geometry_reproduces_failed_no_combo"]
        assert (
            result["controls_exact_to_original"]
            and not result["geometry_quality_passed"]
        )
    else:
        assert result["status"] == "failed"
