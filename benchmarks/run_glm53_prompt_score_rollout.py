# SPDX-License-Identifier: Apache-2.0
"""Fixed 1/3/1 fresh-cache production-policy rollout; never replace a start."""

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from benchmarks.analyze_glm53_deterministic_serving import prefill, timing
from benchmarks.analyze_glm53_quality_pair import compare, compare_observations, require
from benchmarks.kernels.check_glm53_kda_gate import gpu_config
from benchmarks.kernels.glm53_geometry_workload import hardware_identity
from benchmarks.kernels.run_glm53_geometry_serving import (
    PREFIXES,
    gpu_processes,
    inventory,
    local_environment,
)
from slimserve.campaign_sources import PATHS, snapshot
from slimserve.rmsnorm_diagnostic import sha

ROOT = Path(__file__).resolve().parents[1]
PROMPT = Path("/home/tiny/.local/scratch/slimserve-glm53/prompt-source.txt")
MODEL = "/raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4"
ORDER = (
    ("control", "0"),
    ("candidate-1", "1"),
    ("candidate-2", "1"),
    ("candidate-3", "1"),
    ("return-control", "0"),
)
FLAG = "SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS"
HISTORICAL_TPS = {"1": 156.791, "8": 578.993, "16": 779.940}
HISTORICAL = "perf/results/2026-09-08/mhc-paired-serving-remainder"
PREVIOUS = ROOT / "perf/results/2026-09-10/prompt-score-serving-v1"
PINNED = {
    HISTORICAL + "/summary.json": (
        "8be4e28b7b1e7263972ab8956e2eca374ba90ca0dfd1a8f8e8c48d0bc1161722"
    ),
    HISTORICAL + "/boot-1/quality.json": (
        "3986623c123a5fbf530d457885a8b523af796e98c2dcc33259fa98d8a96526c5"
    ),
    HISTORICAL + "/boot-2/quality.json": (
        "058b46bb0facbfd62b90b49529cfe97af85712b73b8b884e0261a389dcc646f1"
    ),
    str((PREVIOUS / "closure.json").relative_to(ROOT)): (
        "27c35be7e4a7bbd5de6b2b2a0d48b8af1a306c38e450590e0ff85e516eb8529b"
    ),
}


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sources():
    names = set(PATHS) | {
        "benchmarks/run_glm53_prompt_score_rollout.py",
        "benchmarks/analyze_glm53_quality_pair.py",
        "benchmarks/analyze_glm53_deterministic_serving.py",
        "slimserve/profiles.json",
        "slimserve/registry.py",
        "slimserve/hardware.py",
        "vllm/compilation/caching.py",
        "vllm/compilation/decorators.py",
        "vllm/config/compilation.py",
        "vllm/env_override.py",
        "perf/glm53-prompt-score-rollout-protocol.md",
        "benchmarks/kernels/check_glm53_kda_gate.py",
        "tests/slimserve/test_prompt_score_rollout.py",
        "tests/slimserve/test_quality_pair_analysis.py",
    }
    return {name: sha(ROOT / name) for name in sorted(names)}


def policy(folder, flag):
    cache = folder / "cache"
    return {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "CUDA_HOME": "/usr/local/cuda-13.0",
        "SLIMSERVE_CACHE": "/raid/weights",
        "OMP_NUM_THREADS": "1",
        "VLLM_CACHE_ROOT": str(cache),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TRITON_CACHE_AUTOTUNING": "1",
        "VLLM_GLM5_MHC_BF16_FN": "1",
        "VLLM_GLM5_MHC_PREFILL_TC": "0",
        "SLIMSERVE_GLM53_NATIVE_ORDER": "0",
        "VLLM_LOGGING_STREAM": "ext://sys.stderr",
        FLAG: flag,
    }


def environment(folder, flag):
    return {
        **{k: v for k, v in os.environ.items() if not k.startswith(PREFIXES)},
        **policy(folder, flag),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def command(folder):
    return [
        str(ROOT / "benchmarks/benchmark_glm53_campaign.py"),
        "--profile",
        "glm53-nvfp4-4",
        "--source",
        str(PROMPT),
        "--output",
        str(folder / "campaign"),
        "--boots",
        "1",
        "--repeats",
        "3",
        "--concurrency",
        "1",
        "8",
        "16",
        "--input-tokens",
        "1000",
        "--output-tokens",
        "300",
        "--cold-prefix",
        "--quality",
        "--prefill",
    ]


def historical_evidence():
    for name, digest in PINNED.items():
        require(sha(ROOT / name) == digest, "completed evidence changed")
    closed = read(ROOT / "perf/results/2026-09-10/prompt-score-serving-v1/closure.json")
    require(
        closed["status"] == "complete" and closed["chunked_quality_passed"],
        "memory-fix serving qualification missing",
    )
    control = closed["cases"][0]
    require(
        control["label"] == "control"
        and sha(PREVIOUS / "control/launch.json") == control["launch_sha256"]
        and sha(PREVIOUS / "control/manifest.json") == control["manifest_sha256"],
        "completed control receipt changed",
    )
    launch = read(PREVIOUS / "control/launch.json")
    require(
        sha(PREVIOUS / "control/campaign/summary.json")
        == launch["files"]["campaign/summary.json"],
        "completed control summary changed",
    )
    manifest = read(PREVIOUS / "control/manifest.json")
    for name in (
        "vllm/v1/sample/prompt_logprobs.py",
        "vllm/v1/sample/sampler.py",
        "vllm/v1/sample/ops/logprobs.py",
    ):
        require(
            sha(ROOT / name) == manifest["sources"][str(ROOT / name)],
            "qualified prompt scoring implementation changed",
        )
    historical = read(ROOT / HISTORICAL / "summary.json")
    require(
        historical["status"] == "complete"
        and historical["environment"].get("SLIMSERVE_GLM53_NATIVE_ORDER", "0") == "0"
        and historical["environment"]["VLLM_GLM5_MHC_BF16_FN"] == "1",
        "historical production-order BF16 reference changed",
    )
    docs = [read(ROOT / HISTORICAL / f"boot-{i}/quality.json") for i in (1, 2)]
    return docs


def freeze():
    require(not git("status", "--short"), "commit before the rollout source freeze")
    # Consume the completed current-binary diagnostic; do not rerun its validator.
    previous = read(PREVIOUS / "control/campaign/summary.json")
    native = previous["runtime"]["native_sha256"]
    require(all(sha(ROOT / n) == h for n, h in native.items()), "native binary changed")
    require(sha(PROMPT) == previous["source_sha256"], "workload source changed")
    return dict(
        git_commit=git("rev-parse", "HEAD"),
        sources=sources(),
        prompt_sha256=sha(PROMPT),
        native_sha256=native,
        runtime=previous["runtime"],
        previous_summary_sha256=sha(PREVIOUS / "control/campaign/summary.json"),
        gpu_config=read(PREVIOUS / "closure.json")["gpu_config_after"],
    )


def check_freeze(frozen):
    require(
        not git("status", "--short")
        and git("rev-parse", "HEAD") == frozen["git_commit"],
        "source tree changed during rollout",
    )
    require(
        sources() == frozen["sources"] and sha(PROMPT) == frozen["prompt_sha256"],
        "frozen source changed",
    )
    require(
        all(sha(ROOT / n) == h for n, h in frozen["native_sha256"].items()),
        "frozen native binary changed",
    )


def expected_plan(folder, flag):
    from slimserve.registry import resolve

    with local_environment(environment(folder, flag)):
        plan = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
        require(
            str(plan.entry_file) == MODEL and not plan.speculative,
            "fixed recipe not selected",
        )
    plan = dataclasses.replace(
        plan,
        engine={
            **plan.engine,
            "enable_prompt_tokens_details": True,
            "enable_per_request_metrics": True,
        },
    )
    # Match the JSON representation recorded by the campaign.
    return json.loads(json.dumps(dataclasses.asdict(plan)))


def audit(folder, flag, frozen, reference):
    summary = read(folder / "campaign/summary.json")
    require(
        summary["status"] == "complete"
        and len(summary["runs"]) == 1
        and summary["git_commit"] == frozen["git_commit"]
        and not summary["git_status"]
        and summary["compatible_profiles"] == ["glm53-nvfp4-4"]
        and summary["command"] == command(folder)
        and summary["environment"] == policy(folder, flag)
        and summary["plan"] == expected_plan(folder, flag)
        and summary["benchmark_implementation_sha256"] == snapshot()
        and summary["source_sha256"] == frozen["prompt_sha256"]
        and summary["diagnostic_only"] is False
        and summary["throughput_is_baseline_eligible"] is True,
        "incomplete or wrong-policy production workload",
    )
    require(
        summary["runtime"]
        == {
            **frozen["runtime"],
            "gpu_before_start": summary["runtime"]["gpu_before_start"],
        }
        and hardware_identity(summary["runtime"]["gpu_before_start"])
        == hardware_identity(frozen["runtime"]["gpu_before_start"]),
        "runtime/hardware changed",
    )
    run = summary["runs"][0]
    require(
        run["status"] == run["teardown"]["status"] == "complete"
        and run["teardown"]["returncode"] == 0
        and run["teardown"]["gpu_release"]["status"] == "complete",
        "teardown failed",
    )
    require(
        run["canaries"]["text"]["answer"] == "4"
        and run["canaries"]["image"]["answer"].lower() == "red",
        "canary failed",
    )
    require(
        len(run["quality_passes"]) == 1
        and run["quality_passes"][0]["status"] == "complete",
        "one complete quality pass per start required",
    )
    document = read(run["quality_path"])
    quality = compare_observations(reference, [document])
    log = (folder / "campaign/boot-1/server.log").read_text()
    warnings = re.findall(r"memory allocation failed with OOM[^\n]*", log)
    record = dict(
        status="audited",
        timing=timing(run, summary),
        prefill=prefill(run, frozen["prompt_sha256"]),
        quality=quality,
        quality_path=run["quality_path"],
        quality_sha256=sha(run["quality_path"]),
        aggregates=summary["aggregates"],
        startup_seconds=run["startup_seconds"],
        quality_seconds=run["quality_passes"][0]["elapsed_seconds"],
        allocation_warnings=warnings,
        teardown=run["teardown"],
    )
    return record, document


def performance_gate(audits):
    controls, candidates = (
        [audits[k] for k in ("control", "return-control")],
        [audits[f"candidate-{i}"] for i in (1, 2, 3)],
    )
    rows = []
    for c in ("1", "8", "16"):
        lower = min(r["aggregates"][c]["median"] for r in controls) * 0.975
        medians = [r["aggregates"][c]["median"] for r in audits.values()]
        rows.append(
            dict(
                concurrency=c,
                minimum_candidate_tps=lower,
                start_medians=medians,
                candidate_nonregression=all(
                    r["aggregates"][c]["median"] >= lower for r in candidates
                ),
                start_spread=max(medians) / min(medians) - 1,
                spread_passed=max(medians) / min(medians) <= 1.025,
                historical_minimum=HISTORICAL_TPS[c] * 0.975,
                historical_floor_passed=all(
                    v >= HISTORICAL_TPS[c] * 0.975 for v in medians
                ),
            )
        )
    cold = []
    for context in ("32768", "131072"):

        def value(row, context=context):
            return row["prefill"]["aggregates"][context][
                "engine_scheduled_to_first_token_ms"
            ]["median"]

        maximum = max(value(r) for r in controls) * 1.025
        cold.append(
            dict(
                context=context,
                maximum_candidate_ms=maximum,
                candidate_medians=[value(r) for r in candidates],
                passed=all(value(r) <= maximum for r in candidates),
            )
        )
    return dict(
        passed=all(
            r["candidate_nonregression"]
            and r["spread_passed"]
            and r["historical_floor_passed"]
            for r in rows
        )
        and all(r["passed"] for r in cold),
        decode=rows,
        prefill=cold,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(
        output.is_relative_to(ROOT / "perf/results") and not output.exists(),
        "new in-repo result directory required",
    )
    reference = historical_evidence()
    frozen = freeze()
    output.mkdir(parents=True)
    record = dict(
        status="running", order=ORDER, frozen=frozen, pinned_evidence=PINNED, runs={}
    )

    def save():
        (output / "rollout.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )

    save()
    try:
        audits, documents = {}, {}
        for label, flag in ORDER:
            check_freeze(frozen)
            require(not gpu_processes().strip(), "another GPU workload active")
            require(gpu_config() == frozen["gpu_config"], "GPU configuration changed")
            for prior in record["runs"].values():
                require(
                    prior["files"] == inventory(Path(prior["folder"])),
                    "completed arm artifacts changed",
                )
            folder = output / label
            folder.mkdir()
            (folder / "cache").mkdir()
            require(not list((folder / "cache").iterdir()), "fresh cache required")
            key = hashlib.sha256(str(output).encode()).hexdigest()[:8]
            unit = f"glm53-prompt-rollout-{key}-{label}"
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
                *command(folder),
            ]
            row = dict(status="starting", folder=str(folder), flag=flag, command=argv)
            record["runs"][label] = row
            save()
            print(
                json.dumps(dict(event="start", label=label, command=argv)), flush=True
            )
            started = time.monotonic()
            with (folder / "serve.log").open("x") as stream:
                try:
                    process = subprocess.run(
                        argv,
                        cwd=ROOT,
                        env=environment(folder, flag),
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
            row.update(
                exit_code=process.returncode,
                seconds=time.monotonic() - started,
                gpu_processes_after=gpu_processes(),
                gpu_config_after=gpu_config(),
            )
            require(
                process.returncode == 0
                and not row["gpu_processes_after"].strip()
                and row["gpu_config_after"] == frozen["gpu_config"],
                "serving/teardown failure; stop series",
            )
            audits[label], documents[label] = audit(folder, flag, frozen, reference)
            (folder / "audit.json").write_text(
                json.dumps(audits[label], indent=2) + "\n"
            )
            row["files"] = inventory(folder)
            require(
                audits[label]["quality"]["passed"],
                "fresh start regressed historical quality; stop series",
            )
            check_freeze(frozen)
            row["status"] = "complete"
            save()
            print(
                json.dumps(
                    dict(
                        event="complete",
                        label=label,
                        tps=audits[label]["aggregates"],
                        allocation_warnings=len(audits[label]["allocation_warnings"]),
                    )
                ),
                flush=True,
            )
        record["quality"] = compare(
            [documents[k] for k in ("control", "return-control")],
            [documents[f"candidate-{i}"] for i in (1, 2, 3)],
        )
        record["performance"] = performance_gate(audits)
        record["allocation_counts"] = {
            k: len(v["allocation_warnings"]) for k, v in audits.items()
        }
        record["allocation_gate"] = all(
            record["allocation_counts"][k] > 0 for k in ("control", "return-control")
        ) and all(record["allocation_counts"][f"candidate-{i}"] == 0 for i in (1, 2, 3))
        require(
            record["quality"]["passed"]
            and record["performance"]["passed"]
            and record["allocation_gate"],
            "rollout gates failed; no default promotion",
        )
        check_freeze(frozen)
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        if record["runs"]:
            row = list(record["runs"].values())[-1]
            if row["status"] != "complete":
                row["status"] = "failed"
            row["files"] = inventory(Path(row["folder"]))
        try:
            record["gpu_processes_at_failure"] = gpu_processes()
            check_freeze(frozen)
            record["failure_source_freeze_verified"] = True
        except Exception as audit_error:
            record["failure_audit_error"] = repr(audit_error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
