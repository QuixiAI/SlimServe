#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real static-future hook on all four RMSNorm sources, no model."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmarks.kernels.check_glm53_cached_rmsnorm import (
    checked_config,
    compare,
    make_inputs,
    run_configuration,
    source_layout,
)
from slimserve import rmsnorm_diagnostic as diagnostic


def load_source(path, kernel_name, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, kernel_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or diagnostic.sha(args.audit) != diagnostic.AUDIT_SHA:
        parser.error("new output and exact source-bound audit required")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        parser.error("GPU workload already active")
    args.output.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(args.output.resolve() / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.output.resolve() / "inductor")
    os.environ["SLIMSERVE_GLM53_NATIVE_ORDER"] = "1"
    os.environ["VLLM_FORCE_AOT_LOAD"] = "1"
    import torch
    import triton
    from torch._inductor.codecache import StaticAutotunerFuture

    audit = json.loads(args.audit.read_text())
    original_namespace = Path(audit["paths"][1]).parent
    original_result = StaticAutotunerFuture.result
    summary = dict(
        status="running",
        checks=[],
        source_sha256=diagnostic.sha(__file__),
        intervention_sha256=diagnostic.sha(diagnostic.__file__),
    )
    output = args.output / "summary.json"

    def save():
        output.write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        for entry in audit["reduction_width_changes"]:
            (source,) = entry["sources"]
            rank = int(source["device_indices"][0])
            filename = Path(source["relative"]).name
            (kernel_name,) = source["kernels"]
            for mode in ("control", "legacy"):
                folder = (args.output / f"rank-{rank}-{mode}").resolve()
                private = folder / "cache" / diagnostic.NAMESPACE
                copied = private / "inductor_cache" / source["relative"]
                copied.parent.mkdir(parents=True)
                shutil.copyfile(
                    original_namespace / "inductor_cache" / source["relative"], copied
                )
                config_path = private / "inductor_cache" / entry["relative"]
                config_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    original_namespace / "inductor_cache" / entry["relative"],
                    config_path,
                )
                target = dict(
                    filename=filename,
                    kernel=kernel_name,
                    source_sha256=source["source_sha256"],
                    configs=entry["configs"],
                    injection_source=str(copied),
                )
                manifest = dict(
                    schema=1,
                    audit_sha256=diagnostic.AUDIT_SHA,
                    namespace=diagnostic.NAMESPACE,
                    cache_root=str(folder / "cache"),
                    original_namespace=str(original_namespace),
                    private_namespace=str(private),
                    receipts=str(folder / "receipts"),
                    targets={str(i): target for i in range(4)},
                )
                path = folder / "manifest.json"
                path.write_text(json.dumps(manifest))
                os.environ[diagnostic.MANIFEST] = str(path)
                os.environ[diagnostic.FLAG] = mode
                os.environ["VLLM_CACHE_ROOT"] = manifest["cache_root"]
                with torch.cuda.device(rank):
                    runner = SimpleNamespace(
                        model_config=SimpleNamespace(
                            hf_text_config=SimpleNamespace(
                                hidden_size=4096, num_hidden_layers=45
                            )
                        ),
                        parallel_config=SimpleNamespace(
                            rank=rank,
                            tensor_parallel_size=4,
                            pipeline_parallel_size=1,
                            enable_expert_parallel=False,
                        ),
                        speculative_config=None,
                        capture_model=lambda: "sealed",
                    )
                    diagnostic.install(runner)
                    bindings = []
                    # The actual AOT load exposed 1/2/2/2 distinct objects for
                    # the four target sources. Qualify every occurrence.
                    for binding in range(1 if rank == 0 else 2):
                        template = load_source(
                            copied, kernel_name, f"rms_smoke_{rank}_{mode}_{binding}"
                        )

                        def compile_config(saved, template=template):
                            cfg = checked_config(saved)
                            return template._precompile_config(
                                triton.Config(
                                    {k: cfg[k] for k in ("XBLOCK", "R0_BLOCK")},
                                    num_warps=cfg["num_warps"],
                                    num_stages=cfg["num_stages"],
                                )
                            )

                        native = compile_config(entry["configs"][1])
                        expected = compile_config(
                            entry["configs"][int(mode == "control")]
                        )
                        template.compile_results = [native]
                        template.configs = None
                        template.filename = str(
                            original_namespace / "inductor_cache" / source["relative"]
                        )
                        future = StaticAutotunerFuture(template)
                        future.reload_kernel_from_src = lambda template=template: (
                            template
                        )
                        rebound = future.result()
                        assert (
                            rebound.launchers[0].cache_hash
                            == expected.make_launcher().cache_hash
                        )
                        bindings.append((rebound, expected))
                    assert runner.capture_model() == "sealed"
                    layout = source_layout(copied.read_text(), kernel_name)
                    weight = torch.ones(4096, dtype=torch.bfloat16)
                    for binding, (rebound, expected) in enumerate(bindings):
                        for rows in (16, 640):
                            x, changed = [
                                make_inputs(rows, seed, 1.0)
                                for seed in (530901, 531001)
                            ]
                            reference = run_configuration(
                                expected.make_launcher(), layout, x, changed, weight
                            )
                            actual = run_configuration(
                                rebound.launchers[0], layout, x, changed, weight
                            )
                            assert all(
                                compare(a, b)["bit_mismatches"] == 0
                                for a, b in zip(reference, actual)
                            )
                            summary["checks"].append(
                                dict(
                                    rank=rank,
                                    mode=mode,
                                    binding=binding,
                                    rows=rows,
                                    exact=True,
                                    hash=rebound.launchers[0].cache_hash,
                                )
                            )
                            save()
                            print(json.dumps(summary["checks"][-1]), flush=True)
                StaticAutotunerFuture.result = original_result
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        StaticAutotunerFuture.result = original_result
        save()


if __name__ == "__main__":
    main()
