# SPDX-License-Identifier: Apache-2.0
"""Bounded no-weights GPU/AOT qualification of RMSNorm geometry isolation.

Trusted local pickle artifacts ONLY. Each prepared rank/mode is attempted once,
in the prescribed order, with audited predecessors. No model forwards or timers.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from benchmarks.kernels.audit_glm53_geometry_graphs import inventory  # noqa: F401
from benchmarks.kernels.check_glm53_attention_norms import require, verify
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from benchmarks.kernels.glm53_artifact_roots import (
    ArtifactRootObserver,
    discover,
    module_inventory,
    read_store,
)
from benchmarks.kernels.glm53_geometry_loader import GeometryLoader
from benchmarks.kernels.glm53_rmsnorm_geometry import SCHEMA
from benchmarks.kernels.prepare_glm53_geometry_loader import QUALIFICATION_SHA
from slimserve import rmsnorm_diagnostic
from slimserve.rmsnorm_diagnostic import NAMESPACE, sha

ORDER = tuple((mode, rank) for mode in ("control", "geometry") for rank in range(4))
MODULE = "benchmarks.kernels.check_glm53_geometry_loader"
AUDITOR_MODULE = "benchmarks.kernels.audit_glm53_geometry_loader"
UNIT_PREFIX = "glm53-geometry-aot-v6"
LOADER = GeometryLoader


def read_json(path):
    return json.loads(Path(path).read_text())


def read_manifest(path):
    path = path.resolve()
    manifest = read_json(path)
    preparation = read_json(path.parent.parent / "preparation.json")
    require(
        manifest["schema"] == SCHEMA
        and manifest["namespace"] == NAMESPACE
        and manifest["qualification_sha256"] == QUALIFICATION_SHA
        and manifest["sources"] == preparation["sources"]
        and preparation["status"] == "prepared"
        and [(r["mode"], r["rank"]) for r in preparation["runs"]] == list(ORDER),
        "wrong prepared geometry series",
    )
    for mode, rank in ORDER:
        row = preparation["runs"][ORDER.index((mode, rank))]
        expected = path.parent.parent / f"{mode}-rank{rank}" / "manifest.json"
        require(
            row["manifest"] == str(expected)
            and sha(expected) == row["manifest_sha256"],
            "prepared manifest changed",
        )
    require(
        path.parent.name == f"{manifest['mode']}-rank{manifest['rank']}"
        and (manifest["mode"], manifest["rank"]) in ORDER
        and manifest["cache_root"] == str(path.parent / "cache")
        and manifest["private_namespace"]
        == str(path.parent / "cache/torch_compile_cache/torch_aot_compile" / NAMESPACE)
        and manifest["receipts"] == str(path.parent / "receipts")
        and [len(manifest["targets"][str(r)]) for r in range(4)] == [3, 3, 3, 4]
        and all(len(manifest["expected_graphs"][str(r)]) == 7 for r in range(4)),
        "wrong rank/mode/private cache or graph/target coverage",
    )
    require(
        all(len(manifest["artifact_roots"][str(r)]) == 7 for r in range(4)),
        "serialized graph root provenance missing",
    )
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        == manifest["git_commit"],
        "source commit changed",
    )
    verify(manifest)
    return manifest, preparation


def prior_receipts(path, manifest, preparation, *, order=ORDER):
    index = order.index((manifest["mode"], manifest["rank"]))
    receipts = []
    for row in preparation["runs"][:index]:
        folder = Path(row["manifest"]).parent
        check_launch(folder, row["manifest_sha256"])
        report = read_json(folder / "run/analysis.json")
        require(
            report["status"] == "complete"
            and report["manifest_sha256"] == row["manifest_sha256"]
            and report["summary_sha256"] == sha(folder / "run/summary.json"),
            "all prescribed predecessors must pass their audits",
        )
        for filename, digest in report["receipts"].items():
            require(sha(filename) == digest, "prior audited receipt changed")
        receipts.append(
            dict(
                mode=row["mode"],
                rank=row["rank"],
                analysis_sha256=sha(folder / "run/analysis.json"),
                launch_sha256=sha(folder / "launch.json"),
            )
        )
    return receipts


def check_launch(folder, manifest_sha):
    record = read_json(folder / "launch.json")
    require(
        record["status"] == "complete"
        and record["manifest_sha256"] == manifest_sha
        and record["load"]["exit_code"] == record["audit"]["exit_code"] == 0
        and not record["gpu_processes_after"].strip(),
        "prescribed load/audit/GPU release did not complete",
    )
    for kind in ("load", "audit"):
        require(
            sha(folder / f"{kind}.log") == record[kind]["log_sha256"],
            "preserved process log changed",
        )
    if "gpu_config_before" in record:
        require(
            record["gpu_config_before"] == record["gpu_config_after"],
            "GPU identity changed",
        )


def environment(manifest):
    for name in (
        rmsnorm_diagnostic.FLAG,
        "TORCHINDUCTOR_DETERMINISTIC",
        "TORCHINDUCTOR_BATCH_INVARIANT",
        "TORCHINDUCTOR_FORCE_FILTER_REDUCTION_CONFIGS",
        "VLLM_BATCH_INVARIANT",
    ):
        require(
            os.getenv(name, "0") in ("", "0"), f"conflicting diagnostic flag: {name}"
        )
    os.environ.update(
        VLLM_CACHE_ROOT=manifest["cache_root"],
        TORCHINDUCTOR_CACHE_DIR=str(
            Path(manifest["private_namespace"]) / "inductor_cache"
        ),
        SLIMSERVE_GLM53_NATIVE_ORDER="1",
        VLLM_FORCE_AOT_LOAD="1",
        VLLM_GLM5_MHC_PREFILL_TC="0",
        VLLM_GLM5_MHC_BF16_FN="1",
    )
    os.environ.pop("TRITON_CACHE_DIR", None)
    rmsnorm_diagnostic.validate_environment()


@contextmanager
def checked_bundles(bundle_class, counts):
    original = bundle_class.__dict__["load_autotuners"]
    lock = threading.RLock()

    def load(cls, tuners):
        loaded = original.__func__(cls, tuners)
        with lock:
            counts.append(dict(expected=len(tuners or []), loaded=len(loaded)))
        require(
            len(loaded) == len(tuners or []),
            "static bundle fallback is not qualification",
        )
        return loaded

    hook = classmethod(load)
    bundle_class.load_autotuners = hook
    try:
        yield
    finally:
        require(
            bundle_class.__dict__["load_autotuners"] is hook,
            "foreign bundle hook change",
        )
        bundle_class.load_autotuners = original


def run(path, *, workflow=None):
    """Shared no-weights lifecycle; workflow supplies manifest and target policy."""
    api = sys.modules[__name__] if workflow is None else workflow
    path = path.resolve()
    manifest, preparation = api.read_manifest(path)
    output = path.parent / "run"
    require(not output.exists(), "preserve prior attempt; no replacement starts")
    priors = api.prior_receipts(path, manifest, preparation)
    require(
        not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "another GPU workload is active",
    )
    private = Path(manifest["private_namespace"])
    require(
        all(sha(private / p) == h for p, h in manifest["original_files"].items()),
        "private cache not identical before first load",
    )
    require(
        all(
            sha(private / p) == h
            for p, h in manifest.get("private_sources", {}).items()
        ),
        "private diagnostic sources changed",
    )
    environment(manifest)
    output.mkdir()
    summary = dict(
        status="running",
        rank=manifest["rank"],
        mode=manifest["mode"],
        manifest_sha256=sha(path),
        source_sha256=sha(api.__file__),
        lifecycle_sha256=sha(__file__),
        prior_receipts=priors,
        model_forward_calls=0,
        weight_tensors_loaded=0,
        static_bundles=[],
    )

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    loader = None
    code_cache = None
    try:
        import torch
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.triton_bundler import TritonBundler

        import vllm._custom_ops  # noqa: F401

        code_cache = PyCodeCache
        rank = manifest["rank"]
        torch.cuda.set_device(rank)
        require(torch.cuda.get_device_capability(rank) == (12, 0), "SM120 required")
        model = private / f"rank_{rank}_0/model"
        summary["model_sha256"] = sha(model)
        summary["torch"] = torch.__version__
        summary["cuda"] = torch.version.cuda
        with (
            (output / "binary-loads.jsonl").open("x") as stream,
            (output / "artifact-roots.jsonl").open("x") as root_stream,
            (output / "loader-events.jsonl").open("x") as loader_stream,
        ):

            def emit(record):
                destination = (
                    stream if record["event"].startswith("binary_") else loader_stream
                )
                destination.write(json.dumps(record, sort_keys=True) + "\n")
                destination.flush()

            loader = api.LOADER(rank, manifest, path, manifest["mode"], emit=emit)

            def emit_root(record):
                root_stream.write(json.dumps(record, sort_keys=True) + "\n")
                root_stream.flush()

            roots = ArtifactRootObserver(
                manifest["artifact_roots"][str(rank)], private, emit=emit_root
            )
            with (
                loader.intercept(),
                roots.intercept(),
                checked_bundles(TritonBundler, summary["static_bundles"]),
            ):
                # Same concurrent deduplicated artifact path as forced AOT serving.
                # Never deserialize the outer model or invoke any forward/capture.
                store, aot_config = read_store(model)
                require(
                    discover(store, private, manifest["expected_graphs"][str(rank)])
                    == manifest["artifact_roots"][str(rank)],
                    "private serialized roots differ from prepared originals",
                )
                summary.update(
                    artifacts=store.num_artifacts(), submodules=store.num_entries()
                )
                require(
                    summary["artifacts"] == 7 and summary["submodules"] == 46,
                    "unexpected cached artifact matrix",
                )
                with torch._functorch.config.patch(aot_config):
                    store.load_all()
                summary["loaded_artifacts"] = len(store.loaded_submodule_store)
                write_new(
                    output / "module-inventory.json",
                    module_inventory(PyCodeCache.modules),
                )
                require(summary["loaded_artifacts"] == 7, "incomplete artifact load")
                root_modules = roots.roots(store, PyCodeCache.modules)
                before = api.inventory(
                    root_modules,
                    manifest,
                    rank,
                    manifest["mode"],
                    loader.observer,
                )
                write_new(output / "graph-bindings.json", before)
                loader.controller.seal(root_modules)
                qualify = getattr(api, "qualify_bindings", None)
                if qualify is not None:
                    summary["weight_tensors_loaded"] = api.LEAF_WEIGHT_COUNT
                    summary["leaf_qualification"] = qualify(
                        root_modules, manifest, before, output
                    )
                # Global sealing is safe HERE: this no-weights gate performs no
                # subsequent capture/forward/non-target compilation.
                loader.observer.seal()
                loader.controller.verify_graphs(root_modules)
                require(
                    api.inventory(
                        roots.roots(store, PyCodeCache.modules),
                        manifest,
                        rank,
                        manifest["mode"],
                        loader.observer,
                    )
                    == before,
                    "graph bindings changed during sealing",
                )
                summary["observed_binary_objects"] = len(loader.observer.records)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        if code_cache is not None and not (output / "module-inventory.json").exists():
            write_new(
                output / "module-inventory.json", module_inventory(code_cache.modules)
            )
        if loader is not None and hasattr(loader, "close"):
            loader.close()
        try:
            verify(manifest)
            summary["original_cache_unchanged"] = True
        except BaseException as error:
            summary.update(
                status="failed",
                freeze_error=repr(error),
                original_cache_unchanged=False,
            )
        save()
        print(json.dumps(summary), flush=True)
    require(summary["status"] == "complete", "AOT loader qualification failed")


def launch(path, *, workflow=None):
    """One bounded attempt plus offline audit; preserve native stdout and status."""
    api = sys.modules[__name__] if workflow is None else workflow
    path = path.resolve()
    manifest, preparation = api.read_manifest(path)
    api.prior_receipts(path, manifest, preparation)
    folder = path.parent
    marker = folder / "launch.json"
    require(not marker.exists(), "this prescribed process was already attempted")
    label = f"{manifest['mode']}-rank{manifest['rank']}"
    record = dict(
        status="running",
        mode=manifest["mode"],
        rank=manifest["rank"],
        manifest_sha256=sha(path),
    )
    if "expected_gpu_config" in manifest:
        from benchmarks.kernels.check_glm53_kda_gate import gpu_config

        record["gpu_config_before"] = gpu_config()
        require(
            record["gpu_config_before"] == manifest["expected_gpu_config"],
            "GPU/driver/power identity changed",
        )
    write_new(marker, record)

    def execute(kind, memory, module, arguments):
        command = [
            "systemd-run",
            "--user",
            "--scope",
            f"--unit={api.UNIT_PREFIX}-{label}-{kind}",
            "-p",
            f"MemoryMax={memory}G",
            "-p",
            "MemorySwapMax=0",
            sys.executable,
            "-m",
            module,
            *arguments,
        ]
        print(
            json.dumps(dict(event="launch", label=label, kind=kind, command=command)),
            flush=True,
        )
        with (folder / f"{kind}.log").open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        record[kind] = dict(
            exit_code=result.returncode,
            command=command,
            log_sha256=sha(folder / f"{kind}.log"),
        )
        marker.write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                dict(event="exit", label=label, kind=kind, code=result.returncode)
            ),
            flush=True,
        )
        return result.returncode

    try:
        run_code = execute(
            "load",
            16,
            api.MODULE,
            ["run", "--manifest", str(path)],
        )
        record["gpu_processes_after"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            text=True,
        )
        audit_code = execute(
            "audit",
            8,
            api.AUDITOR_MODULE,
            ["audit", str(path)],
        )
        if "expected_gpu_config" in manifest:
            record["gpu_config_after"] = gpu_config()
            require(
                record["gpu_config_before"] == record["gpu_config_after"],
                "GPU/driver/power changed during load",
            )
        require(
            run_code == audit_code == 0 and not record["gpu_processes_after"].strip(),
            "load/audit/release gate failed; stop this series",
        )
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        marker.write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "launch"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    (run if args.action == "run" else launch)(args.manifest)


if __name__ == "__main__":
    main()
