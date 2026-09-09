# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def bench(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "glm53_b12x", directory / "benchmark_glm53_b12x.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(bench, tmp_path):
    model = tmp_path / "model" / "snapshots" / bench.REVISION
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("source")
    return argparse.Namespace(
        model_cache=tmp_path / "model",
        jit_cache=cache,
        reference_tokenizer=tmp_path / "reference",
        source=source,
        output=tmp_path / "run",
        boots=3,
        repeats=3,
        startup_timeout=1200,
        diagnostic_nccl=False,
        host_cuda_driver=False,
    )


def test_launcher_keeps_supported_collectives_and_exact_image(bench, tmp_path):
    args = settings(bench, tmp_path)
    argv = bench.docker_create_args(args, "owned-name", 8123)
    assert "--entrypoint" not in argv
    assert "--rm" not in argv  # Exit/OOM receipt before removal.
    assert bench.IMAGE in argv
    assert argv[argv.index("--memory") + 1] == "150g"
    assert argv[argv.index("--memory-swap") + 1] == "150g"
    env = [argv[i + 1] for i, value in enumerate(argv) if value == "-e"]
    assert "NCCL_P2P_DISABLE=0" in env
    assert "B12X_PCIE_ALLREDUCE=1" in env
    assert "MTP_DEPTH=0" in env
    assert "CACHE_MODE=vram" in env
    assert "KV_CACHE_QUANT=fp8_ds_mla" in env
    assert "--disable-custom-all-reduce" not in argv
    mounts = [argv[i + 1] for i, value in enumerate(argv) if value == "--mount"]
    assert len(mounts) == 2
    assert mounts[0].endswith("dst=/model,readonly")
    args.jit_cache = args.model_cache
    with pytest.raises(ValueError, match="separate"):
        bench.docker_create_args(args, "owned-name", 8123)


def test_nccl_diagnostic_only_adds_logging(bench, tmp_path):
    args = settings(bench, tmp_path)
    baseline = bench.docker_create_args(args, "owned-name", 8123)
    args.diagnostic_nccl = True
    diagnostic = bench.docker_create_args(args, "owned-name", 8123)
    index = diagnostic.index("NCCL_DEBUG=INFO")
    assert diagnostic[index - 1] == "-e"
    assert diagnostic[: index - 1] + diagnostic[index + 1 :] == baseline


def test_host_driver_only_suppresses_shell_hook(bench, tmp_path):
    args = settings(bench, tmp_path)
    baseline = bench.docker_create_args(args, "owned-name", 8123)
    args.host_cuda_driver = True
    adapted = bench.docker_create_args(args, "owned-name", 8123)
    index = adapted.index("BASH_ENV=/dev/null")
    assert adapted[index - 1] == "-e"
    assert adapted[: index - 1] + adapted[index + 1 :] == baseline


@pytest.mark.parametrize(
    "fail_kind", ["none", "workload", "startup", "busy", "interrupt"]
)
def test_every_prescribed_start_retained_and_only_owned_ids_removed(
    bench, monkeypatch, tmp_path, fail_kind
):
    args = settings(bench, tmp_path)
    lock = b"source lock\n"
    monkeypatch.setattr(bench, "LOCK_SHA256", hashlib.sha256(lock).hexdigest())
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: lock)
    monkeypatch.setattr(bench, "tokenizer_receipt", lambda *a: (None, {}))
    calls, created, removed, states = [], [], [], {}

    def command(*argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ("git", "rev-parse"):
            return "commit"
        if argv[:2] == ("git", "status"):
            return ""
        if argv[0] == "nvidia-smi":
            return "unrelated-pid" if fail_kind == "busy" else ""
        if argv[:3] == ("docker", "image", "inspect"):
            return "[{}]"
        if argv[:2] == ("docker", "create"):
            identity = str(len(created) + 1) * 64
            created.append(identity)
            states[identity] = {"Running": True, "OOMKilled": False, "ExitCode": 0}
            return identity
        if argv[:2] == ("docker", "inspect"):
            return json.dumps([{"State": states[argv[-1]]}])
        if argv[:2] == ("docker", "stop"):
            states[argv[-1]]["Running"] = False
            return argv[-1]
        if argv[:2] == ("docker", "rm"):
            removed.append(argv[-1])
            assert not states[argv[-1]]["Running"]
            return argv[-1]
        pytest.fail(f"unexpected command: {argv}")

    monkeypatch.setattr(bench, "command", command)

    class Process:
        returncode = 1

        def poll(self):
            return 1 if fail_kind == "startup" else None

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(
        bench.urllib.request,
        "urlopen",
        lambda url, **k: io.BytesIO(
            b"{}"
            if url.endswith("health")
            else json.dumps({"data": [{"id": bench.MODEL_NAME}]}).encode()
        ),
    )
    workloads = []

    def workload(options):
        workloads.append(options)
        if fail_kind == "interrupt":
            raise KeyboardInterrupt("operator interruption")
        if fail_kind == "workload" and len(workloads) == 1:
            raise ValueError("retained workload failure")
        return {"aggregates": {}, "quality": {}, "prefill": {}}

    monkeypatch.setattr(bench, "run_workload", workload)
    if fail_kind == "interrupt":
        with pytest.raises(KeyboardInterrupt, match="operator interruption"):
            bench.run(args)
        receipt = json.loads((args.output / "summary.json").read_text())
        assert receipt["status"] == "failed"
        assert len(receipt["runs"]) == 1
        assert receipt["runs"][0]["status"] == "failed"
        assert removed == created and len(created) == 1
        return
    receipt = bench.run(args)
    saved = json.loads((args.output / "summary.json").read_text())
    assert saved == receipt
    assert len(receipt["runs"]) == 3
    assert receipt["status"] == ("complete" if fail_kind == "none" else "failed")
    assert removed == created
    assert len(created) == (0 if fail_kind == "busy" else 3)
    if fail_kind == "workload":
        assert [r["status"] for r in receipt["runs"]] == [
            "failed",
            "complete",
            "complete",
        ]
    if fail_kind in ("none", "workload"):
        assert len(workloads) == 3
        assert all(w.quality and w.prefill and w.repeats == 3 for w in workloads)
