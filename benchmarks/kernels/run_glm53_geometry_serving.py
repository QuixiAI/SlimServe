# SPDX-License-Identifier: Apache-2.0
"""One attempt per prepared control/geometry/return-control, with frozen evidence.

Serving uses the real profile in a 150 GiB/no-swap scope; offline audit uses 8 GiB.
The parent must also run in an 8 GiB/no-swap scope. No retry, build or cache reuse.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import require, verify
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from benchmarks.kernels.glm53_geometry_workload import (
    MODEL,
    ROOT,
    command,
    expected_environment,
    read,
)
from slimserve.glm53_serving_diagnostic import cases
from slimserve.glm53_serving_diagnostic import policy as serving_policy
from slimserve.rmsnorm_diagnostic import sha

PREFIXES = (
    "VLLM_",
    "SLIMSERVE_",
    "NCCL_",
    "CUDA_",
    "OMP_",
    "TORCHINDUCTOR_",
    "TRITON_",
)


def environment(path, manifest, *, cpu=False):
    # A fixed experiment environment, not an inherited shell's performance knobs.
    result = {k: v for k, v in os.environ.items() if not k.startswith(PREFIXES)}
    result.update(expected_environment(path, manifest))
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    if cpu:
        result["CUDA_VISIBLE_DEVICES"] = ""
    return result


@contextmanager
def local_environment(values):
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def frozen_manifest(path):
    manifest = read(path)
    require(
        not subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).strip(),
        "commit all changes before the serving freeze",
    )
    with local_environment(environment(path, manifest, cpu=True)):
        checked, checked_path = serving_policy(manifest).read_manifest()
    require(checked_path == path and checked == manifest, "prepared manifest differs")
    return manifest


def gpu_processes():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    )


def release_query(record, key):
    try:
        record[key] = gpu_processes()
        return not record[key].strip()
    except Exception as error:
        record[key] = None
        record[key + "_error"] = repr(error)
        return False


def record_gpu_config(manifest, record, key):
    if "expected_gpu_config" in manifest:
        from benchmarks.kernels.check_glm53_kda_gate import gpu_config

        record[key] = gpu_config()
        require(
            record[key] == manifest["expected_gpu_config"],
            "GPU identity/driver/power changed",
        )


def inventory(folder):
    files = {}
    for path in sorted(folder.rglob("*")):
        require(not path.is_symlink(), f"unexpected case alias: {path}")
        if path.is_file() and path != folder / "launch.json":
            files[str(path.relative_to(folder))] = sha(path)
    return files


def check_completed(path):
    folder = path.parent
    record = read(folder / "launch.json")
    manifest = read(path)
    if "expected_gpu_config" in manifest:
        require(
            record["preflight"]["gpu_config_before"]
            == record["gpu_config_after"]
            == manifest["expected_gpu_config"],
            "completed case GPU configuration changed",
        )
    require(
        record["status"] == "complete"
        and record["manifest_sha256"] == sha(path)
        and record["serve"]["exit_code"] == record["audit"]["exit_code"] == 0
        and not record["gpu_processes_after"].strip()
        and not record["gpu_processes_after_audit"].strip(),
        "predecessor serve/audit/release failed; series is terminal",
    )
    require(record["files"] == inventory(folder), "preserved predecessor files changed")
    audit = read(folder / "workload-analysis.json")
    workers = read(folder / "worker-analysis.json")
    require(
        audit["status"] == workers["status"] == "complete"
        and audit["diagnostic_gate_passed"] is True
        and audit["manifest_sha256"] == workers["manifest_sha256"] == sha(path)
        and audit["production_qualified"] is False,
        "predecessor diagnostics incomplete",
    )
    for kind in ("serve", "audit"):
        require(
            sha(folder / f"{kind}.log") == record[kind]["log_sha256"],
            "process log changed",
        )
    return dict(
        label=record["label"],
        manifest_sha256=sha(path),
        launch_sha256=sha(folder / "launch.json"),
        workload_analysis_sha256=sha(folder / "workload-analysis.json"),
        worker_analysis_sha256=sha(folder / "worker-analysis.json"),
    )


def preflight(path, manifest):
    order = cases(manifest)
    index = [label for label, _ in order].index(manifest["label"])
    priors = [
        check_completed(path.parent.parent / label / "manifest.json")
        for label, _ in order[:index]
    ]
    require(
        not any(
            (path.parent / name).exists()
            for name in ("campaign", "receipts", "worker-receipts")
        ),
        "preserve prior serving attempt",
    )
    private = Path(manifest["private_namespace"])
    require(
        inventory(private)
        == {**manifest["original_files"], **manifest.get("private_sources", {})},
        "fresh private cache changed",
    )
    active = gpu_processes()
    require(not active.strip(), "another GPU workload is active")
    from slimserve import registry

    with local_environment(environment(path, manifest, cpu=True)):
        plan = registry.resolve("glm53-nvfp4-4", "rtx6000", 4, None)
        serving_policy(manifest).validate_plan(plan)
        require(
            plan.entry_file == MODEL
            and MODEL.is_dir()
            and manifest["workload"]["model"] == str(MODEL),
            "established recipe model directory not selected",
        )
    record = dict(priors=priors, gpu_processes_before=active, model=str(MODEL))
    record_gpu_config(manifest, record, "gpu_config_before")
    return record


def execute(kind, path, manifest, record):
    cpu = kind == "audit"
    unit_key = hashlib.sha256(str(path.parent.parent).encode()).hexdigest()[:10]
    candidate = cases(manifest)[1][0]
    child = (
        ["-m", "benchmarks.kernels.glm53_geometry_workload", str(path)]
        if cpu
        else command(path, manifest)
    )
    argv = [
        "systemd-run",
        "--user",
        "--scope",
        f"--unit=glm53-{candidate}-{unit_key}-{manifest['label']}-{kind}",
        "-p",
        f"MemoryMax={8 if cpu else 150}G",
        "-p",
        "MemorySwapMax=0",
        str(ROOT / ".venv/bin/python"),
        *child,
    ]
    logfile = path.parent / f"{kind}.log"
    started = time.monotonic()
    record[kind] = dict(status="running", command=argv)
    save(path.parent / "launch.json", record)
    print(json.dumps(dict(event="launch", kind=kind, command=argv)), flush=True)
    with logfile.open("x") as stream:
        try:
            process = subprocess.run(
                argv,
                cwd=ROOT,
                env=environment(path, manifest, cpu=cpu),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        except BaseException as error:
            record[kind].update(status="failed", error=repr(error))
            # Stop only this prepared case's uniquely named scope on interruption.
            stop = [
                "systemctl",
                "--user",
                "stop",
                argv[3].removeprefix("--unit=") + ".scope",
            ]
            try:
                stopped = subprocess.run(
                    stop, stdout=stream, stderr=subprocess.STDOUT, timeout=30
                )
                record[kind]["interrupted_scope_stop"] = dict(
                    command=stop, exit_code=stopped.returncode
                )
            finally:
                stream.flush()
                record[kind]["log_sha256"] = sha(logfile)
                save(path.parent / "launch.json", record)
            raise
    record[kind].update(
        status="exited",
        exit_code=process.returncode,
        seconds=time.monotonic() - started,
        log_sha256=sha(logfile),
    )
    save(path.parent / "launch.json", record)
    print(
        json.dumps(dict(event="exit", kind=kind, code=process.returncode)), flush=True
    )
    return process.returncode


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def launch(path):
    path = path.resolve()
    require(not (path.parent.parent / "closure.json").exists(), "series already closed")
    manifest = read(path)
    marker = path.parent / "launch.json"
    record = dict(status="running", label=manifest["label"], manifest_sha256=sha(path))
    write_new(marker, record)  # A failed preflight is terminal too; never replace it.
    try:
        manifest = frozen_manifest(path)
        record["preflight"] = preflight(path, manifest)
        save(marker, record)
        serve_code = execute("serve", path, manifest, record)
        released = release_query(record, "gpu_processes_after")
        save(marker, record)
        # Still audit failed/partial work. GPU query failure is not a free GPU.
        audit_code = execute("audit", path, manifest, record)
        released_after_audit = release_query(record, "gpu_processes_after_audit")
        record_gpu_config(manifest, record, "gpu_config_after")
        verify(manifest)
        require(
            serve_code == audit_code == 0 and released and released_after_audit,
            "serve/audit/release failed; remaining cases must not launch",
        )
        require(frozen_manifest(path) == manifest, "serving source freeze changed")
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        if "serve" in record and "audit" not in record:
            # Exceptions/interruption are terminal too. Retain an offline audit
            # after the execute helper has stopped this case's owned scope.
            try:
                release_query(record, "gpu_processes_after")
                execute("audit", path, manifest, record)
                release_query(record, "gpu_processes_after_audit")
            except BaseException as audit_error:
                record["exception_audit_error"] = repr(audit_error)
        raise
    finally:
        try:
            record["files"] = inventory(path.parent)
        except BaseException as error:
            record.update(status="failed", inventory_error=repr(error))
        save(marker, record)
    require(record["status"] == "complete", "terminal case inventory failure")


def close_series(series):
    """Close either a complete sequence or the first terminal failure, no reruns."""
    from benchmarks.kernels.glm53_geometry_serving import stable_bindings

    series = series.resolve()
    output = series / "closure.json"
    require(not output.exists(), "preserve prior series closure")
    result = dict(
        status="failed",
        cases=[],
        production_qualified=False,
        auditor_sha256=sha(__file__),
    )
    try:
        result["gpu_processes_after"] = gpu_processes()
        require(not result["gpu_processes_after"].strip(), "GPU release not proven")
        failed = False
        order = cases(read(series / "control/manifest.json"))
        candidate_label = order[1][0]
        for label, _ in order:
            path = series / label / "manifest.json"
            manifest = frozen_manifest(path)
            require(cases(manifest) == order, "mixed diagnostic schemas in one series")
            record_gpu_config(manifest, result, "gpu_config_after")
            marker = path.parent / "launch.json"
            if failed:
                require(not marker.exists(), "a case launched after terminal failure")
                result["cases"].append(dict(label=label, status="unlaunched"))
            elif marker.exists() and read(marker)["status"] == "failed":
                record = read(marker)
                require(
                    record["files"] == inventory(path.parent),
                    "failed case files changed",
                )
                result["cases"].append(
                    dict(label=label, status="failed", launch_sha256=sha(marker))
                )
                failed = True
            else:
                result["cases"].append(dict(status="complete", **check_completed(path)))
        if failed:
            result["status"] = "terminal-failure"
        else:
            audits = {
                label: read(series / label / "workload-analysis.json")
                for label, _ in order
            }
            for label in (candidate_label, "return-control"):
                require(
                    audits[label]["prefill_prompt_sha256"]
                    == audits["control"]["prefill_prompt_sha256"],
                    "cold prefill prompt IDs changed across starts",
                )
                for rank in range(4):
                    name = f"worker-receipts/rank-{rank}/capture-after-bindings.json"
                    control = stable_bindings(read(series / "control" / name))
                    candidate = stable_bindings(read(series / label / name))
                    require(
                        [r for r in control if not r["target"]]
                        == [r for r in candidate if not r["target"]],
                        "non-target choices changed between actual model starts",
                    )
                    if label == "return-control":
                        require(
                            control == candidate,
                            "return-control kernel choices changed",
                        )
            result.update(
                status="complete",
                non_target_exact=True,
                **{
                    f"{candidate_label}_reproduces_failed_no_combo": audits[
                        candidate_label
                    ]["historical_comparisons"]["failed-no-combo"]["exact"],
                    f"{candidate_label}_quality_passed": audits[candidate_label][
                        "quality_passed"
                    ],
                },
                controls_exact_to_original=all(
                    audits[label]["historical_comparisons"]["control"]["exact"]
                    for label in ("control", "return-control")
                ),
                diagnostic_tps={
                    label: a["diagnostic_tps"] for label, a in audits.items()
                },
            )
        result["source_receipts_verified"] = len(manifest["sources"])
        result["original_files_verified"] = len(manifest["original_files"])
    except Exception as error:
        result["error"] = repr(error)
    write_new(output, result)
    print(json.dumps(result), flush=True)
    return result["status"] in ("complete", "terminal-failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "close"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "launch":
        launch(args.path)
    else:
        raise SystemExit(0 if close_series(args.path) else 1)


if __name__ == "__main__":
    main()
