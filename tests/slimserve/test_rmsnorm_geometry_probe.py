# SPDX-License-Identifier: Apache-2.0
import base64
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.kernels import check_glm53_rmsnorm_geometry as probe


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def metric(rows, error=0):
    return dict(
        elements=rows * 4096,
        bit_mismatches=int(error > 0),
        numeric_mismatches=int(error > 0),
        affected_rows=int(error > 0),
        max_bf16_ulp=error,
        max_abs=error * 0.01,
        mean_abs=error * 0.001,
        rms=error * 0.002,
    )


def fixture():
    manifest = dict(
        targets=[dict(rank=r) for r in [0] * 3 + [1] * 3 + [2] * 3 + [3] * 4],
        weight_sha256=[digest("weight")],
    )
    checks = []
    for index, rows, seed, site in probe.case_keys(manifest["targets"]):
        references = [digest("original-output"), digest("changed-output")]
        checks.append(
            dict(
                source_index=index,
                rank=manifest["targets"][index]["rank"],
                rows=rows,
                seed=seed,
                site=site,
                input_sha256=digest("input"),
                changed_sha256=digest("changed-input"),
                oracle_sha256=references,
                repeat_graph_guards_mutation_pass=True,
                oracle={arm: [metric(rows), metric(rows)] for arm in probe.ARMS},
                outputs={arm: list(references) for arm in probe.ARMS},
                cross_config=[metric(rows), metric(rows)],
                passed=True,
            )
        )
    summary = dict(
        status="complete",
        weights=manifest["weight_sha256"],
        frozen_sources_verified=True,
        checks=checks,
    )
    return manifest, summary


def test_full_prescribed_matrix_passes_offline():
    manifest, summary = fixture()
    report = probe.analyze_summary(summary, manifest)
    assert report == dict(
        numerical_pass=True,
        pairs=312,
        oracle_max_bf16_ulp=dict(control=0, geometry=0),
        pairwise_bit_mismatches=0,
        failed_cases=[],
    )


def test_numerical_failure_keeps_complete_matrix_and_fails_verdict():
    manifest, summary = fixture()
    row = summary["checks"][0]
    row["oracle"]["geometry"][0] = metric(row["rows"], 2)
    row["cross_config"][0] = metric(row["rows"], 2)
    row["outputs"]["geometry"][0] = digest("different")
    row["passed"] = False
    summary["status"] = "failed"
    report = probe.analyze_summary(summary, manifest)
    assert report["pairs"] == 312 and not report["numerical_pass"]
    assert len(report["failed_cases"]) == 1
    assert report["oracle_max_bf16_ulp"]["geometry"] == 2
    summary["status"] = "complete"
    with pytest.raises(ValueError, match="terminal verdict"):
        probe.analyze_summary(summary, manifest)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "order",
        "rank",
        "weight",
        "freeze",
        "guard",
        "extent",
        "oracle_hash",
        "paired_hash",
        "phase",
        "arm",
        "nan",
        "negative",
        "equality_count",
        "fake_pass",
        "unchanged_input",
        "unchanged_output",
    ],
)
def test_bad_numerical_receipts_are_rejected(change):
    manifest, summary = fixture()
    row = summary["checks"][0]
    if change == "missing":
        summary["checks"].pop()
    elif change == "order":
        summary["checks"].reverse()
    elif change == "rank":
        row["rank"] = 3
    elif change == "weight":
        summary["weights"] = []
    elif change == "freeze":
        summary["frozen_sources_verified"] = False
    elif change == "guard":
        row["repeat_graph_guards_mutation_pass"] = False
    elif change == "extent":
        row["oracle"]["control"][0]["elements"] = 1536
    elif change == "oracle_hash":
        row["outputs"]["control"][0] = digest("wrong")
    elif change == "paired_hash":
        row["cross_config"][0] = metric(row["rows"], 1)
    elif change == "phase":
        row["outputs"]["geometry"].pop()
    elif change == "arm":
        row["oracle"].pop("geometry")
    elif change == "nan":
        row["oracle"]["geometry"][0]["rms"] = float("nan")
    elif change == "negative":
        row["oracle"]["geometry"][0]["numeric_mismatches"] = -1
    elif change == "equality_count":
        row["oracle"]["geometry"][0]["max_bf16_ulp"] = 1
    elif change == "fake_pass":
        row["passed"] = False
    elif change == "unchanged_input":
        row["changed_sha256"] = row["input_sha256"]
    else:
        row["outputs"]["control"][1] = row["outputs"]["control"][0]
    with pytest.raises(ValueError):
        probe.analyze_summary(summary, manifest)


def binary_fixture(tmp_path):
    target = dict(rank=2, kernel="triton_norm")
    encoded = base64.b32encode(bytes.fromhex(digest("cache"))).decode().rstrip("=")
    selected = dict(hash=encoded, config=probe.GEOMETRY)
    root = probe.rank_cache(tmp_path, 2) / encoded
    root.mkdir(parents=True)
    cubin = root / "triton_norm.cubin"
    cubin.write_bytes(b"full binary including debug data")
    metadata = dict(hash=digest("cache"), name="triton_norm", num_warps=8, num_stages=1)
    path = root / "triton_norm.json"
    path.write_text(json.dumps(metadata))
    return target, selected, path, cubin


def test_binary_receipt_checks_metadata_and_whole_bytes(tmp_path):
    target, selected, path, cubin = binary_fixture(tmp_path)
    original = probe.binary_files(tmp_path, target, selected)
    assert original["cubin_sha256"] == probe.sha(cubin)
    cubin.write_bytes(b"different debug bytes")
    assert probe.binary_files(tmp_path, target, selected) != original
    metadata = json.loads(path.read_text())
    metadata["num_warps"] = 16
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata/config/key"):
        probe.binary_files(tmp_path, target, selected)


@pytest.mark.parametrize("change", ["hash", "kernel", "stages", "key"])
def test_binary_receipt_rejects_wrong_source_metadata(tmp_path, change):
    target, selected, path, _ = binary_fixture(tmp_path)
    metadata = json.loads(path.read_text())
    if change == "hash":
        metadata["hash"] = digest("different")
    elif change == "kernel":
        metadata["name"] = "another_kernel"
    elif change == "stages":
        metadata["num_stages"] = 2
    else:
        selected["hash"] = "../other-cache"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        probe.binary_files(tmp_path, target, selected)


@pytest.mark.parametrize(
    "field",
    [
        "weights",
        "checks",
        "binaries",
        "torch",
        "triton",
        "cuda",
        "git_commit",
        "manifest_sha256",
    ],
)
def test_independent_process_comparison_is_exact(field):
    a = dict.fromkeys(
        (
            "weights",
            "checks",
            "binaries",
            "torch",
            "triton",
            "cuda",
            "git_commit",
            "manifest_sha256",
        ),
        "same",
    )
    b = copy.deepcopy(a)
    probe.compare_processes(a, b)
    b[field] = "different"
    with pytest.raises(ValueError, match=field):
        probe.compare_processes(a, b)


def test_output_is_never_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    probe.write_new(path, dict(status="first"))
    with pytest.raises(FileExistsError):
        probe.write_new(path, dict(status="second"))
    assert json.loads(path.read_text())["status"] == "first"


def test_real_static_cuda_adapter_consumes_bytes_only_after_observation(tmp_path):
    from torch._inductor.runtime.static_triton_launcher import (
        StaticallyLaunchedCudaKernel,
    )

    from benchmarks.kernels.glm53_rmsnorm_geometry import binary_sha

    # Exercise the installed Python lifecycle, replacing only the driver call.
    # No GPU initialization or launch is needed to test this API contract.
    binary = b"complete cubin including debug image"
    path = tmp_path / "kernel.cubin"
    path.write_bytes(binary)
    kernel = StaticallyLaunchedCudaKernel.__new__(StaticallyLaunchedCudaKernel)
    kernel.cubin_raw, kernel.cubin_path = binary, str(path)
    kernel.name, kernel.shared = "test_norm", 0
    kernel.function = kernel.module = None
    calls = []

    def load(filename, name, shared, device):
        assert (filename, name, shared, device) == (str(path), "test_norm", 0, 0)
        assert path.read_bytes() == binary
        calls.append("driver-load")
        return 101, 102, 32, 0

    kernel.C_impl = SimpleNamespace(_load_kernel=load, _unload_kernel=lambda _: None)
    compiled = SimpleNamespace(kernel=kernel)
    expected = hashlib.sha256(binary).hexdigest()
    assert binary_sha(compiled) == expected

    def make_launcher():
        kernel.load_kernel(0)
        return "launcher"

    compiled.make_launcher = make_launcher

    def precompile(config):
        assert config.kwargs == dict(XBLOCK=1, R0_BLOCK=1024)
        assert config.num_warps == 8 and config.num_stages == 1
        return compiled

    template = SimpleNamespace(_precompile_config=precompile)
    assert probe.compile_recorded(template, probe.GEOMETRY) == ("launcher", expected)
    assert calls == ["driver-load"]
    assert kernel.cubin_raw is None and kernel.cubin_path is None
    with pytest.raises(ValueError, match="before launcher load"):
        binary_sha(compiled)
    kernel.close()


def test_in_memory_binary_reader_rejects_ambiguous_or_missing_images():
    from benchmarks.kernels.glm53_rmsnorm_geometry import binary_sha

    kernel = SimpleNamespace(asm=dict(cubin=b"first"), cubin_raw=b"second")
    with pytest.raises(ValueError, match="conflicting"):
        binary_sha(SimpleNamespace(kernel=kernel))
    kernel = SimpleNamespace(cubin_raw=None, cubin_path="/not-a-fallback.cubin")
    with pytest.raises(ValueError, match="unavailable"):
        binary_sha(SimpleNamespace(kernel=kernel))


@pytest.mark.parametrize("change", ["incomplete", "failed", "summary", "manifest"])
def test_process_b_refuses_unqualified_a(tmp_path, change):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}")
    report = dict(
        status="complete",
        numerical_pass=True,
        summary_sha256=probe.sha(summary_path),
        manifest_sha256=probe.sha(manifest_path),
    )
    if change == "incomplete":
        report["status"] = "running"
    elif change == "failed":
        report["numerical_pass"] = False
    elif change == "summary":
        report["summary_sha256"] = "wrong"
    else:
        report["manifest_sha256"] = "wrong"
    (tmp_path / "analysis.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="must pass before B"):
        probe.prior_a(dict(outputs=dict(a=str(tmp_path))), manifest_path)
