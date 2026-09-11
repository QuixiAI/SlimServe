# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "failure", [None, "timeout", "timeout-zero", "interrupt", "success"]
)
def test_fixed_arms_keep_failures_and_only_remove_owned_containers(
    monkeypatch, tmp_path, failure
):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "shell_compat", directory / "benchmark_cuda_shell_compat.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args = argparse.Namespace(output=tmp_path / "results", jit_cache=tmp_path)
    created, removed, stopped = [], [], []

    def command(*argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return ""
        if argv[:2] == ("docker", "create"):
            created.append(argv)
            return str(len(created)) * 64
        if argv[:2] == ("docker", "inspect"):
            return "[{}]"
        if argv[:2] == ("docker", "rm"):
            assert argv[-1] in stopped
            removed.append(argv[-1])
            return argv[-1]
        pytest.fail(f"unexpected command {argv}")

    def stop(identity):
        stopped.append(identity)
        return {"after_stop": {"Running": False, "OOMKilled": False}}

    class Process:
        waited = False

        def wait(self, timeout):
            if len(created) == 1 and not self.waited:
                self.waited = True
                if failure in ("timeout", "timeout-zero"):
                    raise subprocess.TimeoutExpired("owned", timeout)
                if failure == "interrupt":
                    raise KeyboardInterrupt("operator stop")
            return (
                0 if failure in ("success", "timeout-zero") or len(created) == 2 else 1
            )

    monkeypatch.setattr(bench, "command", command)
    monkeypatch.setattr(bench, "stop_container", stop)
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **k: Process())
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            bench.run(args)
    else:
        returned = bench.run(args)
    result = json.loads((args.output / "summary.json").read_text())
    assert len(created) == (1 if failure == "interrupt" else 3)
    assert removed == [str(i) * 64 for i in range(1, len(created) + 1)]
    for index, argv in enumerate(created):
        assert "NCCL_P2P_DISABLE=0" in argv
        assert "NCCL_P2P_LEVEL=SYS" in argv
        assert ("BASH_ENV=/dev/null" in argv) == (index == 1)
        assert argv[argv.index("--memory") + 1] == "16g"
        assert argv[argv.index("--memory-swap") + 1] == "16g"
    if failure == "interrupt":
        assert result["status"] == "failed"
        assert result["arms"][0]["status"] == "failed"
    else:
        assert returned == result
        assert result["status"] == ("complete" if failure == "success" else "failed")
        expected = (
            ["complete"] * 3
            if failure == "success"
            else ["failed", "complete", "complete"]
            if failure == "timeout-zero"
            else ["failed", "complete", "failed"]
        )
        assert [r["status"] for r in result["arms"]] == expected
        if failure in ("timeout", "timeout-zero"):
            assert "150-second" in result["arms"][0]["error"]

    for status, exit_code in (("complete", 0), ("failed", 1)):
        monkeypatch.setattr(bench, "run", lambda args: {"status": status})
        assert (
            bench.main(["--output", str(tmp_path), "--jit-cache", str(tmp_path)])
            == exit_code
        )
