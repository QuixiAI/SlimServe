# SPDX-License-Identifier: Apache-2.0
"""One fresh production-order start with same-live-logit scoring diagnostics.

This is not a revised quality gate or a rollout. Historical quality is reported,
including failures, but the only new criterion is exact same-input scorer parity.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from benchmarks import run_glm53_prompt_score_rollout as rollout
from benchmarks.analyze_glm53_prompt_score_shadow import audit as shadow_audit
from benchmarks.analyze_glm53_quality_pair import compare_observations, require
from slimserve.prompt_score_shadow import FLAG

ROOT = rollout.ROOT
EXTRA_SOURCES = (
    "benchmarks/run_glm53_prompt_score_shadow.py",
    "benchmarks/analyze_glm53_prompt_score_shadow.py",
    "tests/slimserve/test_prompt_score_shadow.py",
    "tests/slimserve/test_prompt_score_shadow_analysis.py",
    "perf/glm53-prompt-score-shadow-protocol.md",
)


def extra_sources():
    return {name: rollout.sha(ROOT / name) for name in EXTRA_SOURCES}


def environment(folder):
    return {**rollout.environment(folder, "1"), FLAG: str(folder / "shadow")}


def audit(folder, frozen, references):
    summary = rollout.read(folder / "campaign/summary.json")
    require(
        summary["status"] == "complete"
        and len(summary["runs"]) == 1
        and summary["git_commit"] == frozen["git_commit"]
        and not summary["git_status"]
        and summary["command"] == rollout.command(folder)
        and summary["compatible_profiles"] == ["glm53-nvfp4-4"]
        and summary["plan"] == rollout.expected_plan(folder, "1")
        and summary["environment"]
        == {**rollout.policy(folder, "1"), FLAG: str(folder / "shadow")}
        and summary["benchmark_implementation_sha256"] == rollout.snapshot()
        and summary["source_sha256"] == frozen["prompt_sha256"]
        and summary["diagnostic_only"] is True
        and summary["throughput_is_baseline_eligible"] is False,
        "wrong/incomplete shadow campaign",
    )
    require(
        summary["runtime"]
        == {
            **frozen["runtime"],
            "gpu_before_start": summary["runtime"]["gpu_before_start"],
        }
        and rollout.hardware_identity(summary["runtime"]["gpu_before_start"])
        == rollout.hardware_identity(frozen["runtime"]["gpu_before_start"]),
        "runtime/hardware differs",
    )
    run = summary["runs"][0]
    require(
        run["status"]
        == run["teardown"]["status"]
        == run["teardown"]["gpu_release"]["status"]
        == "complete"
        and run["teardown"]["returncode"] == 0,
        "serving teardown incomplete",
    )
    require(
        run["canaries"]["text"]["answer"] == "4"
        and run["canaries"]["image"]["answer"].lower() == "red",
        "canary failed",
    )
    require(
        len(run["quality_passes"]) == 1
        and run["quality_passes"][0]["status"] == "complete",
        "one full quality pass required",
    )
    quality = rollout.read(run["quality_path"])
    result = dict(
        status="complete",
        shadow=shadow_audit(folder / "shadow", quality, frozen["sources"]),
        historical_quality=compare_observations(references, [quality]),
        quality_sha256=rollout.sha(run["quality_path"]),
        timing=rollout.timing(run, summary),
        prefill=rollout.prefill(run, frozen["prompt_sha256"]),
        diagnostic_aggregates=summary["aggregates"],
        startup_seconds=run["startup_seconds"],
        quality_seconds=run["quality_passes"][0]["elapsed_seconds"],
        teardown=run["teardown"],
        allocation_warnings=re.findall(
            r"memory allocation failed with OOM[^\n]*",
            (folder / "campaign/boot-1/server.log").read_text(),
        ),
        promotion=False,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    require(
        folder.is_relative_to(ROOT / "perf/results") and not folder.exists(),
        "new result directory required",
    )
    references = rollout.historical_evidence()
    frozen, extra = rollout.freeze(), extra_sources()
    require(not rollout.gpu_processes().strip(), "GPU workload active")
    require(rollout.gpu_config() == frozen["gpu_config"], "GPU configuration changed")
    folder.mkdir(parents=True)
    (folder / "cache").mkdir()
    key = hashlib.sha256(str(folder).encode()).hexdigest()[:8]
    unit = f"glm53-prompt-shadow-{key}"
    argv = [
        "systemd-run",
        "--user",
        "--scope",
        f"--unit={unit}",
        "-p",
        "MemoryMax=150G",
        "-p",
        "MemorySwapMax=0",
        str(ROOT / ".venv/bin/python"),
        *rollout.command(folder),
    ]
    result = dict(
        status="running",
        frozen=frozen,
        extra_sources=extra,
        command=argv,
        environment=environment(folder),
        promotion=False,
    )

    def save():
        (folder / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )

    def check_freeze():
        rollout.check_freeze(frozen)
        require(extra_sources() == extra, "shadow controller/auditor changed")

    save()
    print(json.dumps(dict(event="start", command=argv)), flush=True)
    try:
        with (folder / "serve.log").open("x") as stream:
            try:
                process = subprocess.run(
                    argv,
                    cwd=ROOT,
                    env=environment(folder),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                subprocess.run(
                    ["systemctl", "--user", "stop", unit + ".scope"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                raise
        result["serve_exit_code"] = process.returncode
        require(process.returncode == 0, "shadow serving failed")
        result["audit"] = audit(folder, frozen, references)
        check_freeze()
        result["source_freeze_verified"] = True
        result["gpu_processes_after"] = rollout.gpu_processes()
        result["gpu_config_after"] = rollout.gpu_config()
        require(
            not result["gpu_processes_after"].strip()
            and result["gpu_config_after"] == frozen["gpu_config"],
            "GPU release/config gate failed",
        )
        result["files"] = {
            p: h for p, h in rollout.inventory(folder).items() if p != "result.json"
        }
        result["status"] = "complete"
        save()
        print(
            json.dumps(
                dict(
                    event="complete",
                    shadow=result["audit"]["shadow"],
                    historical_quality_passed=result["audit"]["historical_quality"][
                        "passed"
                    ],
                    promotion=False,
                )
            ),
            flush=True,
        )
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        try:
            check_freeze()
            result["source_freeze_verified"] = True
            result["gpu_processes_after"] = rollout.gpu_processes()
            result["files"] = {
                p: h for p, h in rollout.inventory(folder).items() if p != "result.json"
            }
        except BaseException as audit_error:
            result["failure_audit_error"] = repr(audit_error)
        save()
        raise


if __name__ == "__main__":
    main()
